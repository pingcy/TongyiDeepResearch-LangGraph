# TongYi DeepResearch - LangGraph

这是一个基于Tongyi-DeepResearch开源项目的LangGraph版本的深度研究助手，采用 ReAct Agent 模式实现多工具协作的深度研究能力。本项目基于Tongyi-DeepResearch-30B-A3B模型构建，提供强大的多轮对话和工具调用能力。

## 功能特性

- � 基于Tongyi-DeepResearch-30B-A3B模型，提供强大的推理和工具调用能力
- �🤖 基于ReAct Agent 范式
- 🔍 集成多种工具：网页搜索、学术搜索、文件解析、Python 代码执行等
- 📊 支持多种文件格式解析（PDF、Office、视频等）
- 🌐 智能网页内容提取和摘要
- 💻 安全的 Python 代码执行环境（可选 SandboxFusion）
- 🎯 Token 计数和限制管理
- 🔄 支持多轮对话和上下文维护

## 快速开始

### 环境配置

1. 克隆仓库：

```bash
git clone https://github.com/pingcy/TongyiDeepResearch-LangGraph.git
cd TongyiDeepResearch-LangGraph
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置环境变量：

```bash
cp .env.example .env
# 编辑 .env 文件，填入您的 API keys 和配置
```

### 必要的环境变量

在 `.env` 文件中配置以下必要参数：

```bash
# 模型路径
MODEL_PATH=/path/to/your/model

# API Keys
API_KEY=your_openai_compatible_api_key
API_BASE=your_api_base_url

# Web Search (可选)
SERPER_KEY_ID=your_serper_key
JINA_API_KEYS=your_jina_key

# Dashscope for file parsing (可选)
DASHSCOPE_API_KEY=your_dashscope_key
```

## 使用方法

### 运行 LangGraph Agent

```bash
cd inference
python run_langgraph_agent.py
```

### 运行 Streamlit UI

```bash
cd inference
bash run_streamlit_ui.sh
```

### 运行 A2A Server/Client

```bash
# 启动服务器
cd inference
python a2a_server.py

# 启动客户端（另一个终端）
python a2a_client.py
```

## 项目结构

```
TongYiDeepResearch/
├── inference/              # 推理和工具实现
│   ├── langgraph_react_agent.py   # LangGraph ReAct Agent
│   ├── react_agent.py             # 基础 ReAct Agent
│   ├── tool_*.py                  # 各种工具实现
│   ├── file_tools/                # 文件处理工具
│   └── eval_data/                 # 评估数据
├── models/                 # 模型文件（不包含在 Git 中）
├── requirements.txt        # Python 依赖
├── .env.example           # 环境变量示例
└── README.md              # 本文件
```

## 可用工具

- **SearchSerperAPI**: 网页搜索
- **GoogleScholarSearch**: 学术搜索
- **VisitURL**: 网页内容提取
- **FileParser**: 多格式文件解析
- **PythonInterpreter**: Python 代码执行

## 配置说明

详细的配置选项请参考 `.env.example` 文件，包括：

- 模型和推理参数（Temperature、Top-P 等）
- API Keys 配置
- 多工作进程配置
- 文件解析服务配置
- Python 沙箱配置

## 注意事项

1. **API Keys 安全**: 请勿将 `.env` 文件提交到版本控制系统
2. **模型文件**: 模型文件较大，不包含在仓库中，需要单独下载
3. **依赖项**: 某些功能需要外部服务（如 Serper、Jina、Dashscope）

## License

请查看 [LICENSE](LICENSE) 文件了解许可证信息。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
