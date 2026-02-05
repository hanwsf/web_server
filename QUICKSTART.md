# 快速开始指南

## 第一步：安装依赖

```bash
pip install -r requirements.txt
```

## 第二步：启动服务器

### 选项A：使用启动脚本（推荐）
```bash
./run.sh
```

### 选项B：直接运行Python
```bash
python3 /home/will/.claude/skills/claude-web-server-llm/scripts/server.py
```

## 第三步：访问Web界面

打开浏览器访问：
```
http://localhost:8085
```

## 基本操作

### 1. 创建会话
Web界面会自动创建一个会话，或者通过API：
```bash
curl -X POST http://localhost:8085/api/session
```

### 2. 发送消息
在Web界面输入消息并按 Shift+Enter 发送

或者通过API：
```bash
curl -X POST http://localhost:8085/api/claude \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你的问题",
    "model": "haiku",
    "kb": "KB"
  }'
```

### 3. 查看可用工具
- 访问 /api/skills 查看所有可用的 Skills
- 访问 /api/agents 查看所有可用的 Agents

## 常用curl命令

### 获取服务器状态
```bash
curl http://localhost:8085/api/status | jq
```

### 列出所有Skills
```bash
curl http://localhost:8085/api/skills | jq
```

### 获取会话信息
```bash
curl http://localhost:8085/api/session/SESSION_ID | jq
```

### 获取消息历史
```bash
curl http://localhost:8085/api/messages/SESSION_ID | jq
```

### 删除会话
```bash
curl -X DELETE http://localhost:8085/api/session/SESSION_ID
```

## 模型选择

Web界面支持以下模型：

| 模型 | 说明 |
|------|------|
| Claude Haiku | 快速响应，适合简单任务 |
| Claude Sonnet | 推荐使用，平衡速度和能力 |
| Claude Opus | 最强大，适合复杂任务 |
| Deepseek | 备选模型 |

## 日志查看

服务器启动时会输出详细日志：
```
[INFO] Claude Web Server LLM 启动...
[INFO] 创建新会话: uuid-xxx
[INFO] [CLAUDE] 调用haiku: 你好...
[INFO] [CLAUDE] ✅ 处理成功 (1234字, 2.5s)
```

## 性能调优

### 增加并发数
编辑 `.env` 文件:
```bash
WORKER_THREADS=16  # 增加到16个线程
MAX_SESSIONS=100   # 增加到100个会话
```

### 增加超时时间
如果Claude需要更长时间处理：
```bash
CLAUDE_TIMEOUT=600  # 增加到10分钟
DOCUMENT_TIMEOUT=900  # 文档生成15分钟
```

## 故障排查

### 1. 端口已被占用
```bash
# 查找占用端口8085的进程
lsof -i :8085

# 杀死进程
kill -9 PID
```

### 2. Claude进程无法启动
```bash
# 检查Claude CLI是否已安装
which claude
claude --version
```

### 3. 消息处理超时
增加超时时间或简化请求内容。默认超时为300秒。

## 更多信息

- 完整API文档: 见 README.md
- 源代码: scripts/server.py
- 配置文件: .env
