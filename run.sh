#!/bin/bash
# Claude Web Server LLM 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Claude Web Server LLM 启动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 项目目录: $PROJECT_DIR"
echo "🔧 Python版本: $(python3 --version)"

# 检查Claude CLI
echo ""
echo "检查 Claude CLI..."
CLAUDE_PATH="claude"
if ! command -v "$CLAUDE_PATH" &> /dev/null; then
    echo "❌ Claude 未安装或未在PATH中"
    echo "请确保已安装 Claude CLI 并添加到 PATH"
    exit 1
fi
echo "✅ Claude 已安装: $($CLAUDE_PATH --version 2>&1 || echo 'version unknown')"

# 检查Python依赖
echo ""
echo "检查 Python 依赖..."
pip install -q -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || {
    echo "安装依赖..."
    pip install -r "$PROJECT_DIR/requirements.txt"
}
echo "✅ 依赖已安装"

# 启动服务器
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 启动服务器在 http://localhost:8085"
echo "❌ 按 Ctrl+C 停止服务器"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$PROJECT_DIR"
python3 scripts/server.py
