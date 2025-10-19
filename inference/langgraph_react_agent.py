"""
LangGraph 实现的 React Agent
此实现保持了原始 react_agent.py 的所有功能，包括：
- Token 计数和限制检查
- 特殊工具调用处理 (<tool_call>, <tool_response>)
- 中间结果的流式输出
- 自定义重试逻辑和错误处理
- A2A 协议支持的流式接口
"""

import json
import json5
import os
import time
import random
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated, AsyncIterable, Any
from datetime import datetime
from pathlib import Path

# 在导入工具之前加载环境变量（工具模块在导入时会读取环境变量）
from dotenv import load_dotenv

# 尝试从项目根目录加载 .env 文件
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR.parent / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✅ 已加载环境变量: {ENV_FILE}")
else:
    # 如果根目录没有 .env，尝试当前目录
    load_dotenv()
    print("⚠️  未找到 .env 文件，使用系统环境变量")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer

from prompt import SYSTEM_PROMPT
from tool_file import FileParser
from tool_scholar import Scholar
from tool_python import PythonInterpreter
from tool_search import Search
from tool_visit import Visit


# 常量定义
OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'
MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))
MAX_TOKENS_LIMIT = 110 * 1024
MAX_TIME_LIMIT = 150 * 60  # 150分钟（秒）

# 初始化工具
TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    """获取今天的日期，格式为 YYYY-MM-DD"""
    return datetime.date.today().strftime("%Y-%m-%d")


class AgentState(TypedDict):
    """LangGraph agent 的状态定义"""
    messages: List[Dict]  # 对话历史
    question: str  # 原始问题
    answer: str  # 标准答案（用于评估,暂时不用）
    prediction: str  # Agent 的预测答案
    termination: str  # 终止原因
    round: int  # 当前轮次
    num_llm_calls_available: int  # 剩余 LLM 调用次数
    start_time: float  # 开始时间戳

    # 工具相关字段
    tool_name: Optional[str]  # 当前工具名称
    tool_args: Optional[Dict]  # 当前工具参数
    tool_result: Optional[str]  # 工具结果或错误消息
    
    # 终止控制字段
    should_terminate: bool  # 是否终止
    termination_reason: Optional[str]  # 终止原因
    need_final_answer: bool  # 是否需要最终答案

class LangGraphReactAgent:
    """
    基于 LangGraph 的 React Agent，支持流式输出和自定义工具处理
    同时支持 A2A 协议的流式响应
    """
    
    # A2A 协议支持的内容类型
    SUPPORTED_CONTENT_TYPES = ['text']
    
    def __init__(self, llm_config: Optional[Dict] = None, model_path: Optional[str] = None):
        """
        初始化 LangGraph React Agent
        
        Args:
            llm_config: LLM 生成配置，如果不提供则从环境变量读取
            model_path: 本地模型路径（用于 tokenizer），如果不提供则从环境变量读取
        """
        # 如果没有提供 llm_config，从环境变量读取
        if llm_config is None:
            llm_config = {
                'temperature': float(os.getenv('TEMPERATURE', '0.85')),
                'top_p': float(os.getenv('TOP_P', '0.95')),
                'presence_penalty': float(os.getenv('PRESENCE_PENALTY', '1.1')),
            }
            print(f"📋 使用默认 LLM 配置: {llm_config}")
        
        # 如果没有提供 model_path，从环境变量读取
        if model_path is None:
            model_path = os.getenv('MODEL_PATH', '../models')
            print(f"📁 使用默认模型路径: {model_path}")
        
        self.llm_config = llm_config
        self.llm_local_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 构建图
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图，包含以下节点：
        1. LLM 生成
        2. 工具执行
        3. 答案提取
        4. 终止检查
        """
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("llm_call", self._llm_call_node)
        workflow.add_node("parse_response", self._parse_response_node)
        workflow.add_node("execute_tool", self._execute_tool_node)
        workflow.add_node("check_termination", self._check_termination_node)
        workflow.add_node("extract_answer", self._extract_answer_node)
        
        # 定义边
        workflow.set_entry_point("llm_call")
        
        workflow.add_edge("llm_call", "parse_response")
        
        workflow.add_conditional_edges(
            "parse_response",
            self._should_execute_tool,
            {
                "execute_tool": "execute_tool",
                "check_termination": "check_termination"
            }
        )
        
        workflow.add_edge("execute_tool", "check_termination")
        
        workflow.add_conditional_edges(
            "check_termination",
            self._should_continue,
            {
                "llm_call": "llm_call",
                "extract_answer": "extract_answer",
                "end": END
            }
        )
        
        workflow.add_edge("extract_answer", END)
        
        return workflow.compile()
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        """计算消息历史中的 token 数量"""
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(full_prompt, return_tensors="pt")
        return len(tokens["input_ids"][0])
    
    def _call_llm_with_retry(self, messages: List[Dict], max_tries: int = 5) -> str:
        """
        调用 LLM API，采用指数退避重试逻辑
        
        Args:
            messages: 对话历史
            max_tries: 最大重试次数
            
        Returns:
            LLM 生成的内容
        """
        
        openai_api_key = os.getenv("API_KEY", "your-openai-api-key")
        openai_api_base = os.getenv("API_BASE", "http://127.0.0.1:1234/v1")

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )
        
        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(f"--- 尝试调用 LLM，第 {attempt + 1}/{max_tries} 次 ---")
                
                chat_response = client.chat.completions.create(
                    model='alibaba-nlp_tongyi-deepresearch-30b-a3b@q5_k_l',
                    messages=messages,
                    stream=True,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_config.get('temperature', 0.6),
                    top_p=self.llm_config.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000
                )
                
                #non-streaming
                #content = chat_response.choices[0].message.content
                
                content = ""
                for chunk in chat_response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end='', flush=True)

                # OpenRouter
                #reasoning_content = "<think>\n" + (
                #   chat_response.choices[0].message.reasoning.strip()
                #   if chat_response.choices[0].message.reasoning
                #   else ""
                #) + "\n</think>"
                #content = reasoning_content + content
                
                if content and content.strip():
                    print("\n--- LLM 调用成功 ---")
                    return content.strip()
                else:
                    print(f"警告: 第 {attempt + 1} 次尝试收到空响应")
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"错误: 第 {attempt + 1} 次尝试失败，API 错误: {e}")
            except Exception as e:
                print(f"错误: 第 {attempt + 1} 次尝试失败，未预期错误: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"等待 {sleep_time:.2f} 秒后重试...")
                time.sleep(sleep_time)
            else:
                print("错误: 所有重试尝试已用尽")
        
        return "vllm server error!!!"
    
    def _execute_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """
        执行工具调用，带有适当的错误处理
        
        Args:
            tool_name: 要执行的工具名称
            tool_args: 工具参数
            
        Returns:
            工具执行结果字符串
        """
        if tool_name not in TOOL_MAP:
            return f"错误: 工具 {tool_name} 未找到"
        
        try:
            tool_args["params"] = tool_args
            
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args.get("files", [])}
                raw_result = asyncio.run(
                    TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus")
                )
                result = str(raw_result) if not isinstance(raw_result, str) else raw_result
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args)
                result = raw_result
                
            return result
        except Exception as e:
            return f"执行工具 {tool_name} 时出错: {str(e)}"
    
    def _llm_call_node(self, state: AgentState) -> AgentState:
        """
        节点: 调用 LLM 生成下一个响应
        此节点通过产生中间结果来处理流式输出
        """
        print(f"\n=== 第 {state['round'] + 1} 轮推理 ===")
        
        # 增加轮次计数器
        state["round"] += 1
        state["num_llm_calls_available"] -= 1
        
        # 调用 LLM
        #print(f"current messages: {state['messages']}")
        content = self._call_llm_with_retry(state["messages"])

        
        # 从内容中移除任何 tool_response 标签
        if '<tool_response>' in content:
            pos = content.find('<tool_response>')
            content = content[:pos]
        
        # 将助手响应添加到消息中
        state["messages"].append({
            "role": "assistant",
            "content": content.strip()
        })
        
        return state
    
    def _parse_response_node(self, state: AgentState) -> AgentState:
        """
        节点: 解析 LLM 响应以提取工具调用或答案
        """
        print("\n=== 解析响应 ===")
        
        content = state["messages"][-1]["content"]
        
        # 检查工具调用
        if '<tool_call>' in content and '</tool_call>' in content:
            tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            
            # 处理 Python 解释器特殊情况
            if "python" in tool_call_str.lower() and "<code>" in tool_call_str:
                try:
                    code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    code_raw = code_raw.split('<code>')[1].split('</code>')[0].strip()
                    state["tool_name"] = "PythonInterpreter"
                    state["tool_args"] = {"code": code_raw}
                    print(f"解析工具: {state['tool_name']}")
                except Exception as e:
                    print(f"Python 解析错误: {e}")
                    state["tool_result"] = "[Python Interpreter Error]: Formatting error."
                    state["tool_name"] = None
                    
            else:
                try:
                    tool_call = json5.loads(tool_call_str)
                    state["tool_name"] = tool_call.get('name', '')
                    state["tool_args"] = tool_call.get('arguments', {})
                    print(f"解析工具: {state['tool_name']}")
                except Exception as e:
                    print(f"JSON 解析错误: {e}")
                    state["tool_result"] = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                    state["tool_name"] = None
        
        return state
    
    def _should_execute_tool(self, state: AgentState) -> str:
        """
        条件边: 确定是否应该执行工具
        """
        if state.get("tool_name"):
            return "execute_tool"
        return "check_termination"
    
    def _execute_tool_node(self, state: AgentState) -> AgentState:
        """
        节点: 执行已解析的工具调用
        """
        tool_name = state.get("tool_name")
        tool_args = state.get("tool_args", {})
        
        print(f"\n=== 执行工具: {tool_name} ===")
        
        if state.get("tool_result"):
            # 错误已在解析节点中设置
            result = state["tool_result"]
        elif tool_name == "PythonInterpreter":
            try:
                result = TOOL_MAP['PythonInterpreter'].call(tool_args.get("code", ""))
            except Exception as e:
                result = f"[Python Interpreter Error]: {str(e)}"
        else:
            result = self._execute_tool_call(tool_name, tool_args)
        
        # 使用 tool_response 标签格式化结果
        result = f"<tool_response>\n{result}\n</tool_response>"
        
        # 将工具结果添加到消息中
        state["messages"].append({
            "role": "user",
            "content": result
        })
        
        # 清理工具相关状态
        state["tool_name"] = None
        state["tool_args"] = None
        state["tool_result"] = None
        
        return state
    
    def _check_termination_node(self, state: AgentState) -> AgentState:
        """
        节点: 检查各种终止条件
        """
        print("\n=== 检查终止条件 ===")
        
        content = state["messages"][-1]["content"]
        
        # 检查是否提供了答案
        if '<answer>' in content and '</answer>' in content:
            state["should_terminate"] = True
            state["termination_reason"] = "answer"
            state["need_final_answer"] = False  
            print("✓ 检测到答案标签")
            return state
        
        # 检查时间限制
        elapsed_time = time.time() - state["start_time"]
        if elapsed_time > MAX_TIME_LIMIT:
            state["should_terminate"] = True
            state["termination_reason"] = "time_limit"
            state["prediction"] = "No answer found after 2h30mins"
            state["termination"] = "No answer found after 2h30mins"
            print(f"✗ 超时: {elapsed_time:.0f}秒")
            return state
        
        # 检查 LLM 调用次数限制
        if state["num_llm_calls_available"] <= 0:
            state["messages"][-1]["content"] = "Sorry, the number of llm calls exceeds the limit."
            state["should_terminate"] = True
            state["termination_reason"] = "llm_call_limit"
            print("✗ LLM 调用次数超限")
            return state
        
        # 检查 token 限制
        token_count = self._count_tokens(state["messages"])
        print(f"Token 数量: {token_count:,}")
        
        if token_count > MAX_TOKENS_LIMIT:
            print(f"✗ Token 超限: {token_count:,} > {MAX_TOKENS_LIMIT:,}")
            
            # 添加最终提示以生成答案
            state["messages"][-1]["content"] = (
                "You have now reached the maximum context length you can handle. "
                "You should stop making tool calls and, based on all the information above, "
                "think again and provide what you consider the most likely answer in the "
                "following format:<think>your final thinking</think>\n<answer>your answer</answer>"
            )
            
            state["should_terminate"] = True
            state["termination_reason"] = "token_limit"
            state["need_final_answer"] = True
            return state
        
        # 如果没有满足终止条件，则继续
        state["should_terminate"] = False
        state["termination_reason"] = None
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        条件边: 基于终止检查确定下一步
        """
        if state.get("should_terminate"):
            # 优先检查是否需要生成最终答案（token 超限的情况）
            if state.get("need_final_answer"):
                return "llm_call"
            # 检测到答案标签，直接提取答案
            elif state.get("termination_reason") == "answer":
                return "extract_answer"
            # LLM 调用次数超限或时间超限，直接结束
            elif state.get("termination_reason") in ["llm_call_limit", "time_limit"]:
                return "end"
            else:
                return "end"
        return "llm_call"
    
    def _extract_answer_node(self, state: AgentState) -> AgentState:
        """
        节点: 从响应中提取最终答案
        """
        print("\n=== 提取最终答案 ===")
        
        content = state["messages"][-1]["content"]
        
        if '<answer>' in content and '</answer>' in content:
            prediction = content.split('<answer>')[1].split('</answer>')[0]
            termination = state.get("termination_reason", "answer")
            
            if state.get("need_final_answer"):
                termination = "generate an answer as token limit reached"
        else:
            prediction = content if state.get("need_final_answer") else "No answer found."
            termination = "format error" if state.get("need_final_answer") else "answer not found"
            
            if state["num_llm_calls_available"] == 0:
                termination = "exceed available llm calls"
        
        state["prediction"] = prediction
        state["termination"] = termination
        
        return state
    
    async def stream(self, query: str, context_id: str) -> AsyncIterable[Dict[str, Any]]:
        """
        流式调用智能体，适配 A2A 协议的状态格式
        
        通过智能体异步流式处理查询，并根据不同执行阶段返回不同状态的响应。
        
        参数:
            query (str): 用户的查询问题
            context_id (str): 对话上下文ID，用于保持会话连续性
            
        返回:
            AsyncIterable[Dict[str, Any]]: 异步迭代器，产生包含状态和内容的字典
            可能的状态包括:
                - "working": Agent 正在思考或执行工具调用
                - "completed": 处理完成，包含最终答案
                - "failed": 处理失败
        
        异常:
            捕获所有异常并以失败状态返回错误信息
        """
        try:
            # 构建 system prompt
            system_prompt = SYSTEM_PROMPT
            cur_date = datetime.now().strftime("%Y-%m-%d")
            system_prompt = system_prompt + str(cur_date)
            
            # 初始化状态，包含 system prompt
            initial_state = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "question": query,
                "answer": "",
                "prediction": "",
                "termination": "",
                "round": 0,
                "num_llm_calls_available": MAX_LLM_CALL_PER_RUN,
                "start_time": time.time(),
                "tool_name": None,
                "tool_args": None,
                "tool_result": None,
                "should_terminate": False,
                "termination_reason": None,
                "need_final_answer": False,
            }
            
            # 配置：设置递归限制和线程ID
            config = {
                "recursion_limit": 150,
                "configurable": {"thread_id": context_id}
            }
            
            # 第一个状态更新：开始处理
            yield {
                "status": "working",
                "content": "开始处理您的请求..."
            }
            
            # 使用 astream 异步流式处理
            async for state in self.graph.astream(initial_state, config=config, stream_mode="values"):

                # 获取当前状态的最新消息
                if "messages" not in state or len(state["messages"]) == 0:
                    continue
                
                latest_message = state["messages"][-1]
                content = latest_message.get("content", "")
                role = latest_message.get("role", "")
                
                # 根据消息类型返回不同状态
                if state.get("prediction"):
                    # 已提取最终答案（优先检查，因为这是最终状态）
                    yield {
                        "status": "completed",
                        "content": state["prediction"]
                    }
                
                elif role == "assistant" and "<tool_call>" in content:
                    # LLM 生成了工具调用
                    tool_name = state.get("tool_name")
                    
                    if tool_name:
                        # Parse 节点已完成，显示具体工具名称
                        yield {
                            "status": "working",
                            "content": f"正在调用工具【{tool_name}】..."
                        }
                    else:
                        # LLM 节点刚完成，还未解析
                        yield {
                            "status": "working",
                            "content": "检测到工具调用，正在解析..."
                        }
                
                elif role == "user" and "<tool_response>" in content:
                    # 工具返回结果
                    yield {
                        "status": "working",
                        "content": "工具执行完成，正在处理结果..."
                    }
                    
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Agent 执行错误: {error_details}")
            yield {
                "status": "failed",
                "content": f"执行过程中发生错误: {str(e)}"
            }