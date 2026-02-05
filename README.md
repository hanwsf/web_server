# Claude Web Server LLM Skill

一个多线程的Web服务器，使用Claude作为后端服务进行语义路由和智能调度。

## 快速开始

### 安装依赖

```bash
pip install requests
```

### 启动服务器

```bash
python3 /home/will/.claude/skills/claude-web-server-llm/scripts/server.py
```

访问：http://localhost:8085

### 系统要求

- Python 3.8+
- requests
- Claude CLI 工具已安装

## 架构设计

```
Web Server (8085)
│
├── 会话管理 (SessionManager)
│   ├── Session 1 ────→ Claude命令执行
│   ├── Session 2 ────→ Claude命令执行
│   └── Session N ────→ Claude命令执行
│
├── 语义调度系统
│   ├── Skills 自动扫描 (~/.claude/skills/)
│   ├── Agents 列表 (Bash, Explore, Plan等)
│   └── LLM语义分析器 (智能路由)
│
├── 多模型后端
│   ├── Claude (opus, sonnet, haiku)
│   ├── NVIDIA API
│   └── Deepseek API
│
└── WebUI 前端
    └── 实时聊天界面
```

## 核心功能

### 语义路由
- 使用LLM分析用户意图
- 智能路由到对应的Skill/Agent/Subagent
- 无法判断时直接调用Claude处理

### 支持的Agent类型
- **Bash**: 命令执行专家
- **Explore**: 代码库探索
- **Plan**: 架构规划
- **general-purpose**: 通用研究代理
- **claude-code-guide**: Claude使用指南

### 支持的模型
- **Claude Opus**: 最强大的推理能力
- **Claude Sonnet**: 推荐使用，性价比高
- **Claude Haiku**: 快速响应
- **Deepseek**: 备选模型
- **NVIDIA API**: 备选模型

## API文档

### Session 管理

#### 创建会话
```
POST /api/session

Response:
{
  "session_id": "uuid",
  "model": "haiku",
  "created_at": "2026-01-27T..."
}
```

#### 获取会话信息
```
GET /api/session/{session_id}

Response:
{
  "session_id": "uuid",
  "model": "haiku",
  "created_at": "2026-01-27T...",
  "last_activity": "2026-01-27T...",
  "message_count": 5
}
```

#### 删除会话
```
DELETE /api/session/{session_id}

Response:
{
  "status": "deleted"
}
```

### 消息交互

#### 发送消息
```
POST /api/claude
Body: {
  "prompt": "用户消息",
  "model": "haiku",
  "kb": "KB"
}

Response:
{
  "response": "Claude的处理结果",
  "intent": {...},
  "scheduler_used": true
}
```

#### 获取消息历史
```
GET /api/messages/{session_id}

Response:
{
  "session_id": "uuid",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."},
    ...
  ]
}
```

### 资源发现

#### 列出Skills
```
GET /api/skills

Response:
{
  "count": 5,
  "skills": [
    {"name": "skill-name", "description": "...", "path": "/path/to/skill"},
    ...
  ]
}
```

#### 列出Agents
```
GET /api/agents

Response:
{
  "count": 4,
  "agents": [
    {"name": "Bash", "description": "...", "subagent_type": "Bash"},
    ...
  ]
}
```

### 服务器状态

#### 获取状态
```
GET /api/status

Response:
{
  "status": "running",
  "sessions": 3,
  "timestamp": "2026-01-27T..."
}
```

## 特性

### 多线程处理
- ThreadPoolExecutor with 16 workers
- 支持最多100个并发会话
- 每个会话独立的Claude执行
- 异步消息处理

### 持久化会话
- 通过Claude --print命令执行
- 自动会话超时清理（1小时）
- 线程安全的消息存储

### 语义调度
- 自动扫描 ~/.claude/skills/ 目录
- 解析SKILL.md元数据
- LLM智能路由决策
- 实时Agents/SubAgents列表

### Web前端
- 实时聊天界面
- 会话管理
- 模型选择（Claude/Deepseek）
- 消息历史

## 配置参数

编辑 `.env` 文件：

```bash
# Claude Configuration
CLAUDE_EXECUTABLE=claude
CLAUDE_MODELS=opus,sonnet,haiku
CLAUDE_TIMEOUT=300

# Server settings
WEB_PORT=8085
MAX_SESSIONS=100
SESSION_TIMEOUT=3600
WORKER_THREADS=16

# Working directory
WORK_DIR=/home/will/Downloads/opencode_p
SKILLS_DIR=/home/will/.claude/skills/
```

## 故障排查

### Claude进程无法启动
```
错误: claude executable not found

解决: 确保claude已安装并在PATH中
$ claude --version
```

### 会话响应超时
```
提示: ⏳ 请求超时，Claude仍在处理

原因: Claude处理复杂请求
解决: 增加超时时间或简化请求
```

### 端口被占用
```
错误: [Errno 98] Address already in use

解决: 修改WEB_PORT或杀死占用进程
$ lsof -i :8085
$ kill -9 <PID>
```

## 语义路由流程

```
用户输入
    ↓
LLM语义分析
    ↓
┌─────────────────────────────────────┐
│ 判断类型                             │
├─────────────────────────────────────┤
│ → Skill匹配 → 执行Skill操作          │
│ → Agent匹配 → 调用Agent处理          │
│ → 无法判断 → 直接调用Claude处理      │
└─────────────────────────────────────┘
    ↓
返回结果给用户
```

## 安全考虑

⚠️ **警告**: 此服务器暴露了Claude的全部能力，包括文件编辑、命令执行等。

建议：
1. 仅在内网使用或添加身份验证
2. 添加请求验证和速率限制
3. 定期审计Claude处理的请求

## 许可

MIT License


创建的 claude-web-server-llm skill 位于 /home/will/.claude/skills/claude-web-server-llm/                                                                       
                                                                                                                                                                   
  主要修改                                                                                                                                                         
  文件: SKILL.md                                                                                                                                                   
  修改内容: 更新名称、描述、配置为 Claude                                                                                                                          
  ────────────────────────────────────────                                                                                                                         
  文件: server.py                                                                                                                                                  
  修改内容: 将 call_opencode 替换为 call_claude，使用 claude --print --model 命令；API端点从 /api/opencode 改为 /api/claude；前端标题和模型选择改为 Claude         
  ────────────────────────────────────────                                                                                                                         
  文件: scheduler.py                                                                                                                                               
  修改内容: 更新模型映射添加 Claude 模型；Agent 从 opencode-code-guide 改为 claude-code-guide；路径改为 ~/.claude/skills/                                          
  ────────────────────────────────────────                                                                                                                         
  文件: .env                                                                                                                                                       
  修改内容: 添加 CLAUDE_EXECUTABLE、CLAUDE_MODELS、CLAUDE_TIMEOUT 配置；更新 SKILLS_DIR                                                                            
  ────────────────────────────────────────                                                                                                                         
  文件: run.sh                                                                                                                                                     
  修改内容: 改为检查 Claude CLI                                                                                                                                    
  ────────────────────────────────────────                                                                                                                         
  文件: README.md                                                                                                                                                  
  修改内容: 完整更新为 Claude 文档                                                                                                                                 
  ────────────────────────────────────────                                                                                                                         
  文件: QUICKSTART.md                                                                                                                                              
  修改内容: 更新快速开始指南                                                                                                                                       
  核心功能差异                                                                                                                                                     
  ┌────────────────────────────┬──────────────────────────────────┐                                                                                                
  │ 原 opencode-web-server-llm │     新 claude-web-server-llm     │                                                                                                
  ├────────────────────────────┼──────────────────────────────────┤                                                                                                
  │ 使用 opencode run 命令     │ 使用 claude --print --model 命令 │                                                                                                
  ├────────────────────────────┼──────────────────────────────────┤                                                                                                
  │ 默认模型 minimax-m2.1      │ 默认模型 haiku                   │                                                                                                
  ├────────────────────────────┼──────────────────────────────────┤                                                                                                
  │ API 端点 /api/opencode     │ API 端点 /api/claude             │                                                                                                
  ├────────────────────────────┼──────────────────────────────────┤                                                                                                
  │ 无降级机制                 │ Claude 失败时自动降级到 Deepseek │                                                                                                
  └────────────────────────────┴──────────────────────────────────┘                                                                                                
  启动方式                                                                                                                                                         
                                                                                                                                                                   
  # 使用启动脚本                                                                                                                                                   
  bash /home/will/.claude/skills/claude-web-server-llm/run.sh                                                                                                      
                                                                                                                                                                   
  # 或直接运行                                                                                                                                                     
  python3 /home/will/.claude/skills/claude-web-server-llm/scripts/server.py                                                                                        
                                                                                                                                                                   
  # 访问                                                                                                                                                           
  http://localhost:8085                                                                                                                                            
                                                                 # web_server
