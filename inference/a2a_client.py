"""
通义深度研究智能体 A2A 客户端
- 支持流式响应
- 支持推送通知
- 友好的中文输出
"""
import asyncio
import logging
import sys
import pathlib
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4
import base64
import os
import threading
import json

# 添加当前目录到Python路径
current_dir = pathlib.Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import asyncclick as click
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    Part,
    SendStreamingMessageRequest,
    SendMessageRequest,
    TextPart,
    FilePart,
    FileWithBytes,
    Task,
    TaskState,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    MessageSendConfiguration,
    JSONRPCErrorResponse,
    GetTaskRequest,
    TaskQueryParams, 
    GetTaskResponse,
    GetTaskSuccessResponse,
    PushNotificationConfig
)
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_message(message: Message) -> None:
    """友好地展示消息对象的详细信息

    Args:
        message: 消息对象
    """
    print("\n" + "=" * 60)
    print("💬 消息信息")
    print("=" * 60)
    print(f"🆔 消息ID: {message.messageId}")
    print(f"📌 上下文ID: {message.contextId}") 
    print(f"🆔 任务ID: {message.taskId}")
    print(f"👤 角色: {message.role}")
    print("📝 消息内容:")
    for part in message.parts:
        if hasattr(part.root, 'text') and part.root.text:
            print(f"  {part.root.text}")
        elif hasattr(part.root, 'file') and isinstance(part.root.file, FileWithBytes):
            file_name = part.root.file.name if part.root.file.name else "未命名文件"
            print(f"  📎 包含文件: {file_name}")


def display_task(task: Task) -> None:
    """友好地展示任务对象的详细信息
    
    Args:
        task: 任务对象
    """
    print("\n" + "=" * 60)
    print("📋 任务信息")
    print("=" * 60)
    
    print(f"🆔 任务ID: {task.id}")
    
    if hasattr(task, 'contextId'):
        print(f"📌 上下文ID: {task.contextId}")
    
    if hasattr(task, 'status') and hasattr(task.status, 'state'):
        state = task.status.state
        state_emoji = {
            "submitted": "📤",
            "working": "⚙️",
            "input-required": "❓",
            "completed": "✅",
            "canceled": "🛑",
            "failed": "❌", 
            "rejected": "⛔",
            "auth-required": "🔒",
        }.get(state, "🔄")
        print(f"{state_emoji} 状态: {state}")

        if hasattr(task.status, 'message') and task.status.message:
            message = task.status.message
            print("\n📝 状态消息:")
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        print(f"  {part.root.text}")
    
    if hasattr(task, 'artifacts') and task.artifacts:
        print("\n📦 工件信息:")
        for artifact in task.artifacts:
            artifact_name = artifact.name if hasattr(artifact, 'name') else "未命名"
            print(f"  工件名称: {artifact_name}")
            if hasattr(artifact, 'parts'):
                print("  工件内容:")
                for part in artifact.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        print(f"    {part.root.text}")


def display_task_status_update(status_event: TaskStatusUpdateEvent) -> None:
    """友好地展示任务状态更新的详细信息
    
    Args:
        status_event: 任务状态更新事件
    """
    print("\n" + "=" * 60)
    print("📊 任务状态更新")
    print("=" * 60)
    
    state = status_event.status.state if hasattr(status_event.status, 'state') else "unknown"
    
    state_emoji = {
        "submitted": "📤",
        "working": "⚙️",
        "input-required": "❓",
        "completed": "✅",
        "canceled": "🛑",
        "failed": "❌", 
        "rejected": "⛔",
        "auth-required": "🔒",
    }.get(state, "🔄")
    
    print(f"{state_emoji} 状态: {state}")
    
    if hasattr(status_event.status, 'message') and status_event.status.message:
        message = status_event.status.message
        if hasattr(message, 'parts'):
            for part in message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    print(f"📝 消息: {part.root.text}")


def display_artifact_update(artifact_event: TaskArtifactUpdateEvent) -> None:
    """友好地展示任务工件更新的详细信息
    
    Args:
        artifact_event: 任务工件更新事件
    """
    print("\n" + "=" * 60)
    print("📦 任务工件更新")
    print("=" * 60)
    
    artifact_name = artifact_event.artifact.name if hasattr(artifact_event.artifact, 'name') else "未命名"
    print(f"📋 工件名称: {artifact_name}")
    
    if hasattr(artifact_event.artifact, 'parts'):
        print("📄 工件内容:")
        for part in artifact_event.artifact.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                print(f"{part.root.text}")


async def query_task(client: A2AClient, task_id: str) -> Optional[Task]:
    """查询任务状态
    
    Args:
        client: A2A客户端
        task_id: 任务ID
    """
    try:
        task_response = await client.get_task(
            GetTaskRequest(
                id=str(uuid4()),
                params=TaskQueryParams(id=task_id),
            )
        )
        if isinstance(task_response.root, JSONRPCErrorResponse):
            logger.error(f"获取任务失败: {task_response.root.error}")
            print(f"\n❌ 发生错误: {task_response.root.error}")
            return None
        
        elif isinstance(task_response.root, GetTaskSuccessResponse):
            print("\n" + "=" * 60 + "\n获取到完整任务信息:\n" + "=" * 60)
            task_result = task_response.root.result
            display_task(task_result)
            return task_result

    except Exception as e:
        logger.error(f"获取完整任务失败: {e}")
        return None


class PushNotificationReceiver:
    """推送通知接收器"""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=lambda loop: loop.run_forever(), args=(self.loop,)
        )
        self.thread.daemon = True
        self.thread.start()

    def start(self):
        """启动推送通知接收服务器"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.start_server(),
                self.loop,
            )
            logger.info('======= 推送通知监听器已启动 =======')
        except Exception as e:
            logger.error(f'启动推送通知监听器失败: {e}')

    async def start_server(self):
        """启动HTTP服务器接收推送通知"""
        self.app = Starlette()
        self.app.add_route(
            '/notify', self.handle_notification, methods=['POST']
        )
        self.app.add_route(
            '/notify', self.handle_validation_check, methods=['GET']
        )

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level='critical'
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()

    async def handle_validation_check(self, request: Request):
        """处理验证请求"""
        validation_token = request.query_params.get('validationToken')
        logger.info(f'收到推送通知验证 => {validation_token}')

        if not validation_token:
            return Response(status_code=400)

        return Response(content=validation_token, status_code=200)

    async def handle_notification(self, request: Request):
        """处理推送通知"""
        data = await request.json()
        logger.info(f'收到推送通知: {data}')
        return Response(status_code=200)


async def complete_task(
    client: A2AClient,
    use_push_notifications: bool = False,
    notification_host: str = "localhost",
    notification_port: int = 5000,
    use_streaming: bool = True,
) -> Tuple[bool]:
    """完成一个智能体任务
    
    Args:
        client: A2A客户端
        use_push_notifications: 是否使用推送通知
        notification_host: 推送通知主机
        notification_port: 推送通知端口
        task_id: 任务ID
        context_id: 上下文ID
        use_streaming: 是否使用流式请求模式
        
    Returns:
        Tuple[继续执行标志, 上下文ID, 任务ID]
    """
    prompt = await click.prompt(
        '\n请输入您想问智能体的问题 (:q 或 quit 退出)',
        default='',
    )
    if prompt.lower() in [':q', 'quit']:
        return False

    print("\n" + "=" * 60)
    print(f"🔹 任务启动") 
    print("=" * 60)

    send_message_payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': prompt}
            ],
            'message_id': uuid4().hex,
        },
    }

    payload = MessageSendParams(**send_message_payload)

    if use_push_notifications:
        payload.configuration.pushNotificationConfig = PushNotificationConfig(
            url=f'http://{notification_host}:{notification_port}/notify',
            authentication={'schemes': ['bearer']}
        )
        
    try:
        logger.info('正在发送请求，等待响应...')
        print("\n" + "=" * 60 + "\n智能体思考中...\n" + "=" * 60)
        
        if use_streaming:
            response_stream = client.send_message_streaming(
                SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
            
            async for result in response_stream:
                if isinstance(result.root, JSONRPCErrorResponse):
                    logger.error(f"错误: {result.root.error}")
                    print(f"\n❌ 发生错误: {result.root.error}")
                    return False
                
                event = result.root.result
                
                if isinstance(event, Task):
                    display_task(event)
                    
                elif isinstance(event, Message):
                    display_message(event)
                
                elif isinstance(event, TaskStatusUpdateEvent):
                    display_task_status_update(event)
                    
                elif isinstance(event, TaskArtifactUpdateEvent):
                    display_artifact_update(event)
        else:
            await client.send_message(
                SendMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
        
        return False
    except Exception as e:
        logger.error(f"处理响应时发生错误: {e}", exc_info=True)
        print(f"\n❌ 发生错误: {str(e)}")
        return True


@click.command()
@click.option('--url', default='http://localhost:10002', help='智能体服务器地址')
@click.option('--push', is_flag=True, default=False, help='是否启用推送通知')
@click.option('--push-host', default='localhost', help='推送通知接收主机')
@click.option('--push-port', default=5099, help='推送通知接收端口')
@click.option('--task-id', default=None, help='要查询的任务ID')
@click.option('--no-streaming', is_flag=True, default=False, help='使用非流式请求模式')
async def main(url: str, push: bool, push_host: str, push_port: int, 
               task_id: Optional[str], no_streaming: bool) -> None:
    """通义深度研究智能体 A2A 客户端"""
    print("\n" + "=" * 60)
    print("📱 通义深度研究智能体客户端")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=600) as httpx_client:
        try:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            
            print("🔍 正在连接智能体...")
            agent_card = await resolver.get_agent_card()
            print("✅ 成功连接智能体")
            print(f"📝 智能体名称: {agent_card.name}")
            print(f"📝 智能体描述: {agent_card.description}")
            
            if push:
                push_receiver = PushNotificationReceiver(
                    host=push_host,
                    port=push_port
                )
                push_receiver.start()
                print(f"📢 已启动推送通知接收器: http://{push_host}:{push_port}/notify")
            
            client = A2AClient(agent_card=agent_card, httpx_client=httpx_client)
            
            if task_id:
                print(f"\n🔍 正在查询任务ID: {task_id}")
                task = await query_task(client, task_id)
                if task:
                    print("\n✅ 任务查询完成")
                else:
                    print("\n❌ 任务查询失败")
                return
            
            continue_loop = True
            context_id = None
            local_task_id = None
            
            while continue_loop:
                continue_loop= await complete_task(
                    client, 
                    push, 
                    push_host, 
                    push_port,
                    use_streaming=not no_streaming
                )
                
        except Exception as e:
            logger.error(f"错误: {e}", exc_info=True)
            print(f"\n❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    main()
