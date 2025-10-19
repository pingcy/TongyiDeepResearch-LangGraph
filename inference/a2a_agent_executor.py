"""
通义深度研究智能体执行器 - 处理A2A请求并执行智能体
"""
import logging
import sys
import pathlib
import os
from typing import Optional

# 添加当前目录到Python路径
current_dir = pathlib.Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from langgraph_react_agent import LangGraphReactAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TongYiDeepResearchAgentExecutor(AgentExecutor):
    """通义深度研究智能体执行器"""

    def __init__(self, llm_config: Optional[dict] = None, model_path: Optional[str] = None):
        """
        初始化执行器
        
        Args:
            llm_config: LLM 配置字典，可选，不提供则使用环境变量
            model_path: 模型路径，可选，不提供则使用环境变量
        """
        # 直接使用 LangGraphReactAgent，它会自动处理默认值
        self.agent = LangGraphReactAgent(
            llm_config=llm_config,
            model_path=model_path
        )
        logger.info(f"✅ TongYiDeepResearchAgentExecutor 初始化完成")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """执行智能体"""
        
        print(f"收到请求，上下文: {context.__dict__}")
        
        # 获取或创建任务
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            logger.info(f"创建新任务: {task.id}")
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # 验证请求
            error = self._validate_request(context)
            if error:
                logger.error("请求参数无效")
                updater.update_status(
                    "failed",
                    new_agent_text_message(
                        "请求参数无效",
                        task.context_id,
                        task.id,
                    ),
                )
                return

            query = context.get_user_input()
            logger.info(f"处理查询: {query}")
            
            # 调用 agent 的 stream 方法
            response = self.agent.stream(query, task.id)
            
            async for item in response:
                status = item.get("status")
                content = item.get("content", "")

                logger.info(f"Agent 状态更新: {status}, 内容: {content[:5000]}...")

                if status == "working":
                    # 更新工作状态
                    await updater.update_status(
                        "working",
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                    
                elif status == "completed":
                    # 任务完成
                    logger.info(f"任务完成: {content[:100]}...")
                    
                    # 添加工件（最终答案）
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name="research_result",
                    )
                    
                    # 标记任务完成
                    await updater.complete(
                        new_agent_text_message(
                            "研究任务完成",
                            task.context_id,
                            task.id,
                        )
                    )
                    break
                    
                elif status == "failed":
                    # 任务失败
                    logger.error(f"任务失败: {content}")
                    await updater.update_status(
                        "failed",
                        new_agent_text_message(
                            f"任务执行失败: {content}",
                            task.context_id,
                            task.id,
                        ),
                    )
                    break

        except Exception as e:
            logger.error(f"处理响应流时发生错误: {e}", exc_info=True)
            try:
                await updater.update_status(
                    "failed",
                    new_agent_text_message(
                        f"执行过程中发生错误: {str(e)}",
                        task.context_id,
                        task.id,
                    ),
                )
            except:
                pass
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """验证请求"""
        # 检查是否有用户输入
        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            logger.warning("请求缺少用户输入")
            return True
        return False

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """取消任务"""
        logger.warning("收到取消任务请求，但当前不支持取消操作")
        raise ServerError(error=UnsupportedOperationError("不支持取消任务"))
