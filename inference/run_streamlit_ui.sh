#!/bin/bash
# 通义深度研究智能体 Streamlit UI 启动脚本

echo "🚀 启动通义深度研究智能体 Streamlit UI"
echo "========================================"
echo ""
echo "请确保 A2A 服务器已经启动："
echo "  python a2a_server.py --host localhost --port 10002"
echo ""
echo "========================================"
echo ""

streamlit run a2a_streamlit_ui.py --server.port 8501
