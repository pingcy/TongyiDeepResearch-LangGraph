"""
通义深度研究智能体服务器入口
"""
import logging
import os
import sys
import pathlib

# 添加当前目录到Python路径
current_dir = pathlib.Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from langgraph_react_agent import LangGraphReactAgent
from a2a_agent_executor import TongYiDeepResearchAgentExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingConfigError(Exception):
    """配置缺失异常"""


@click.command()
@click.option('--host', 'host', default='localhost', help='服务器主机地址')
@click.option('--port', 'port', default=10002, help='服务器端口')
def main(host, port):
    """启动通义深度研究智能体服务器"""
    try:
        # LangGraphReactAgent 会自动从环境变量读取配置
        # 不需要显式传递 llm_config 和 model_path
        
        # 定义 Agent 能力
        capabilities = AgentCapabilities(
            streaming=True, 
            pushNotifications=True
        )
        
        # 定义 Agent 技能
        skill = AgentSkill(
            id='deep_research',
            name='深度研究',
            description='使用多种工具进行深度研究和分析，包括网络搜索、文献检索、文件解析、Python代码执行等',
            tags=['研究', '分析', '搜索', '编程', '文献'],
            examples=[
                '帮我研究一下量子计算的最新进展',
                '分析这个数据集并给出统计报告',
                '搜索并总结关于深度学习的最新论文'
            ],
        )
        
        # 创建 Agent Card
        agent_card = AgentCard(
            name='TongyiDeepResearchAgent',
            description='通义深度研究智能体，能够执行复杂的研究任务，包括网络搜索、学术文献检索、文件解析和Python代码执行等功能',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=LangGraphReactAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=LangGraphReactAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # 创建HTTP客户端
        httpx_client = httpx.AsyncClient()
        
        # 创建请求处理器（使用默认配置，从环境变量读取）
        request_handler = DefaultRequestHandler(
            agent_executor=TongYiDeepResearchAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        
        # 创建服务器应用
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )

        logger.info("=" * 60)
        logger.info("通义深度研究智能体服务器")
        logger.info("=" * 60)
        logger.info(f"配置来源: 环境变量 (.env 文件)")
        logger.info(f"服务地址: http://{host}:{port}/")
        logger.info(f"Agent Card: http://{host}:{port}/.well-known/agent.json")
        logger.info("=" * 60)
        
        # 启动服务器
        uvicorn.run(server.build(), host=host, port=port)

    except MissingConfigError as e:
        logger.error(f'配置错误: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'服务器启动时发生错误: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
