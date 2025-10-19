#!/usr/bin/env python3
"""
简单测试脚本 - 验证 LangGraphReactAgent 的基本功能
"""
import asyncio

# langgraph_react_agent 会自动加载环境变量
from langgraph_react_agent import LangGraphReactAgent

async def test_agent():
    """测试 Agent 的 stream 方法"""
    print("=" * 60)
    print("测试 LangGraphReactAgent")
    print("=" * 60)
    
    # 创建 Agent（使用默认配置，自动从环境变量读取）
    print("\n创建 Agent...")
    agent = LangGraphReactAgent()
    
    # 如果需要自定义配置，可以这样：
    # agent = LangGraphReactAgent(
    #     llm_config={'temperature': 0.6, 'top_p': 0.95},
    #     model_path='../models'
    # )
    
    # 测试问题
    query = "介绍下阿里巴巴的tongyi-deepresearch语言模型"
    context_id = "test-context-001"
    
    print(f"\n测试问题: {query}")
    print(f"上下文ID: {context_id}")
    print("\n开始执行...\n")
    
    # 调用 stream 方法
    try:
        async for item in agent.stream(query, context_id):
            status = item.get("status")
            content = item.get("content", "")
            print(f"[{status}] {content[:5000]}")
            
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())

