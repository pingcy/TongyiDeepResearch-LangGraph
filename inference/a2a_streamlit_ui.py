"""
通义深度研究智能体 Streamlit UI
- 基于 A2A 协议的对话式界面
- 实时显示任务状态和进度
- Markdown 格式展示结果
"""
import asyncio
import logging
import sys
import pathlib
from typing import Optional, List, Dict, Any
from uuid import uuid4
import json

# 添加当前目录到Python路径
current_dir = pathlib.Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    Part,
    SendStreamingMessageRequest,
    TextPart,
    Task,
    TaskState,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    MessageSendConfiguration,
    JSONRPCErrorResponse,
)
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 页面配置
st.set_page_config(
    page_title="通义深度研究智能体",
    page_icon="🤖",
    layout="wide",
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_id" not in st.session_state:
    st.session_state.context_id = None
if "agent_card" not in st.session_state:
    st.session_state.agent_card = None
if "client" not in st.session_state:
    st.session_state.client = None
if "httpx_client" not in st.session_state:
    st.session_state.httpx_client = None


def format_task_state(state: str) -> str:
    """格式化任务状态为带表情符号的文本"""
    state_map = {
        "submitted": "📤 已提交",
        "working": "⚙️ 工作中",
        "input-required": "❓ 需要输入",
        "completed": "✅ 已完成",
        "canceled": "🛑 已取消",
        "failed": "❌ 失败",
        "rejected": "⛔ 已拒绝",
        "auth-required": "🔒 需要认证",
    }
    return state_map.get(state, f"🔄 {state}")


def extract_text_from_parts(parts: List[Part]) -> str:
    """从 Part 列表中提取文本"""
    texts = []
    for part in parts:
        if hasattr(part.root, 'text') and part.root.text:
            texts.append(part.root.text)
    return "\n".join(texts) if texts else ""


async def initialize_client(agent_url: str):
    """初始化 A2A 客户端"""
    try:
        # 创建一个持久的 httpx 客户端（不使用 async with，保持打开状态）
        httpx_client = httpx.AsyncClient(timeout=600)
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_url)
        agent_card = await resolver.get_agent_card()
        return agent_card, httpx_client
    except Exception as e:
        logger.error(f"初始化客户端失败: {e}")
        raise


async def send_message_and_stream(
    client: A2AClient,
    user_input: str,
    context_id: Optional[str] = None,
    status_placeholder = None,
    progress_container = None,
):
    """发送消息并处理流式响应，实时更新 UI"""
    
    # 创建消息
    message = Message(
        role='user',
        parts=[TextPart(text=user_input)],
        messageId=str(uuid4())
    )
    
    # 创建请求参数
    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            acceptedOutputModes=['text'],
        ),
    )
    
    # 存储结果
    result_data = {
        "status": "submitted",
        "status_messages": [],
        "artifacts": [],
        "final_message": None,
        "error": None,
    }
    
    try:
        # 发送流式请求
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                result_data["error"] = str(result.root.error)
                result_data["status"] = "failed"
                if status_placeholder:
                    status_placeholder.caption(f"状态: {format_task_state('failed')}")
                break
            
            event = result.root.result
            
            # 处理任务事件
            if isinstance(event, Task):
                if hasattr(event, 'status') and hasattr(event.status, 'state'):
                    result_data["status"] = event.status.state
                    # 实时更新状态
                    if status_placeholder:
                        status_placeholder.caption(f"状态: {format_task_state(event.status.state)}")
                    
                if hasattr(event, 'artifacts') and event.artifacts:
                    result_data["artifacts"] = event.artifacts
            
            # 处理消息事件
            elif isinstance(event, Message):
                text = extract_text_from_parts(event.parts)
                if text:
                    result_data["final_message"] = text
            
            # 处理状态更新事件
            elif isinstance(event, TaskStatusUpdateEvent):
                if hasattr(event.status, 'state'):
                    result_data["status"] = event.status.state
                    # 实时更新状态
                    if status_placeholder:
                        status_placeholder.caption(f"状态: {format_task_state(event.status.state)}")
                    
                if hasattr(event.status, 'message') and event.status.message:
                    text = extract_text_from_parts(event.status.message.parts)
                    if text:
                        status_msg = {
                            "state": event.status.state,
                            "message": text,
                        }
                        result_data["status_messages"].append(status_msg)
                        
                        # 实时显示进度消息
                        if progress_container:
                            state_text = format_task_state(event.status.state)
                            # 累积显示所有进度消息
                            progress_html = ""
                            for msg in result_data["status_messages"]:
                                msg_state = format_task_state(msg["state"])
                                progress_html += f"**{msg_state}**\n\n{msg['message']}\n\n---\n\n"
                            progress_container.markdown(progress_html)
            
            # 处理工件更新事件
            elif isinstance(event, TaskArtifactUpdateEvent):
                artifact_text = extract_text_from_parts(event.artifact.parts)
                artifact_name = event.artifact.name if hasattr(event.artifact, 'name') else "结果"
                artifact_data = {
                    "name": artifact_name,
                    "content": artifact_text,
                }
                result_data["artifacts"].append(artifact_data)
                
                # 实时显示工件（追加到进度消息中）
                if progress_container:
                    # 更新进度区域，包含工件信息
                    progress_html = ""
                    for msg in result_data["status_messages"]:
                        msg_state = format_task_state(msg["state"])
                        progress_html += f"**{msg_state}**\n\n{msg['message']}\n\n---\n\n"
                    
                    # 添加工件预览
                    if result_data["artifacts"]:
                        progress_html += f"\n\n### 📦 收到工件\n\n"
                        for art in result_data["artifacts"]:
                            progress_html += f"- {art['name']}\n"
                    
                    progress_container.markdown(progress_html)
                
    except Exception as e:
        logger.error(f"处理流式响应时出错: {e}", exc_info=True)
        result_data["error"] = str(e)
        result_data["status"] = "failed"
        if status_placeholder:
            status_placeholder.caption(f"状态: {format_task_state('failed')}")
    
    return result_data


def render_message(message: Dict[str, Any], index: int):
    """渲染单条消息"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # 显示状态信息
            if "status" in message:
                status_text = format_task_state(message["status"])
                st.caption(f"状态: {status_text}")
            
            # 显示中间状态消息
            if "status_messages" in message and message["status_messages"]:
                with st.expander("📊 任务进度详情", expanded=False):
                    for status_msg in message["status_messages"]:
                        state_text = format_task_state(status_msg["state"])
                        st.markdown(f"**{state_text}**")
                        st.markdown(status_msg["message"])
                        st.divider()
            
            # 显示最终消息
            if "final_message" in message and message["final_message"]:
                st.markdown("### 💬 响应消息")
                st.markdown(message["final_message"])
            
            # 显示工件（结果）
            if "artifacts" in message and message["artifacts"]:
                st.markdown("### 📦 研究结果")
                for artifact in message["artifacts"]:
                    artifact_name = artifact.get("name", "结果")
                    artifact_content = artifact.get("content", "")
                    
                    with st.expander(f"📄 {artifact_name}", expanded=True):
                        st.markdown(artifact_content)
            
            # 显示错误
            if "error" in message and message["error"]:
                st.error(f"❌ 错误: {message['error']}")


async def main_async():
    """异步主函数"""
    
    # 标题
    st.title("🤖 通义深度研究智能体")
    st.caption("基于 A2A 协议的智能研究助手")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        agent_url = st.text_input(
            "Agent 服务器地址",
            value="http://localhost:10002",
            help="A2A Agent 服务器的 URL"
        )
        
        if st.button("🔄 重新连接"):
            # 关闭旧的 httpx 客户端
            if st.session_state.httpx_client is not None:
                try:
                    await st.session_state.httpx_client.aclose()
                except:
                    pass
            st.session_state.agent_card = None
            st.session_state.client = None
            st.session_state.httpx_client = None
            st.session_state.messages = []
            st.session_state.context_id = None
            st.rerun()
        
        st.divider()
        
        # 初始化客户端
        if st.session_state.agent_card is None:
            with st.spinner("正在连接智能体..."):
                try:
                    # 创建持久的 httpx 客户端
                    agent_card, httpx_client = await initialize_client(agent_url)
                    st.session_state.agent_card = agent_card
                    st.session_state.httpx_client = httpx_client
                    st.session_state.client = A2AClient(
                        agent_card=agent_card,
                        httpx_client=httpx_client
                    )
                    st.success("✅ 已连接到智能体")
                except Exception as e:
                    st.error(f"❌ 连接失败: {e}")
                    return
        
        # 显示 Agent 信息
        if st.session_state.agent_card:
            st.markdown("### 📝 智能体信息")
            st.markdown(f"**名称**: {st.session_state.agent_card.name}")
            st.markdown(f"**描述**: {st.session_state.agent_card.description}")
            
            if hasattr(st.session_state.agent_card, 'skills') and st.session_state.agent_card.skills:
                st.markdown("**技能**:")
                for skill in st.session_state.agent_card.skills:
                    st.markdown(f"- {skill.name}: {skill.description}")
    
    # 主聊天区域
    if st.session_state.client is None:
        st.warning("⚠️ 请先在侧边栏配置并连接到智能体服务器")
        return
    
    # 显示历史消息
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, idx)
    
    # 用户输入
    user_input = st.chat_input("请输入您的研究问题...")
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # 显示用户消息
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # 显示助手响应（带实时状态更新）
        with st.chat_message("assistant", avatar="🤖"):
            # 使用 st.status 来显示实时进度
            with st.status("正在处理您的请求...", expanded=True) as status:
                status_text = st.empty()
                progress_area = st.empty()
                
                status_text.caption(f"状态: {format_task_state('submitted')}")
                
                # 创建进度消息容器
                progress_messages = []
                
                # 发送消息并实时获取响应
                result = await send_message_and_stream(
                    st.session_state.client,
                    user_input,
                    st.session_state.context_id,
                    status_placeholder=status_text,
                    progress_container=progress_area,
                )
                
                # 更新最终状态
                if result["status"] == "completed":
                    status.update(label="✅ 任务完成！", state="complete", expanded=False)
                elif result["error"]:
                    status.update(label="❌ 任务失败", state="error", expanded=True)
                else:
                    status.update(label=f"📊 任务状态: {format_task_state(result['status'])}", state="running", expanded=False)
            
            # 显示最终结果
            if result["final_message"]:
                st.markdown("### 💬 响应消息")
                st.markdown(result["final_message"])
            
            # 显示工件
            if result["artifacts"]:
                st.markdown("### 📦 研究结果")
                for artifact in result["artifacts"]:
                    artifact_name = artifact.get("name", "结果")
                    artifact_content = artifact.get("content", "")
                    with st.expander(f"📄 {artifact_name}", expanded=True):
                        st.markdown(artifact_content)
            
            # 显示错误
            if result["error"]:
                st.error(f"❌ 错误: {result['error']}")
            
            # 构建助手消息
            assistant_message = {
                "role": "assistant",
                "content": "",
                "status": result["status"],
                "status_messages": result["status_messages"],
                "final_message": result["final_message"],
                "artifacts": result["artifacts"],
                "error": result["error"],
            }
            
            # 添加到消息历史
            st.session_state.messages.append(assistant_message)
            
            # 重新渲染以显示完整结果
            st.rerun()


def main():
    """主函数入口"""
    try:
        asyncio.run(main_async())
    except Exception as e:
        st.error(f"应用程序错误: {e}")
        logger.error(f"应用程序错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
