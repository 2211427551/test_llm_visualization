# Transformer计算模拟器 - 后端服务 (扩展版)

支持MoE、稀疏注意力、流式传输和可视化的Transformer计算模拟器后端服务。

## 功能特性

- 🚀 基于FastAPI的高性能异步API
- 🔢 使用NumPy实现完整的Transformer计算流程
- 📊 逐步追踪和记录所有中间计算结果
- 🎯 支持多会话并发处理
- 📝 自动生成的API文档
- 🧠 **MoE (Mixture of Experts) 支持** - 可配置专家数量和top-k选择
- 🔍 **稀疏注意力模式** - 滑窗、全局+局部、分块、随机等多种模式
- 🌊 **流式传输** - SSE和WebSocket支持实时数据推送
- 📦 **数据分块** - 支持大负载的分页加载
- 🎨 **可视化配置** - 2D/3D/图形/时间轴渲染模式
- ⚡ **性能优化** - 可配置精度、GZip压缩、缓存机制

## 技术栈

- Python 3.9+
- FastAPI - Web框架
- NumPy - 矩阵计算
- Pydantic - 数据验证
- Uvicorn - ASGI服务器

## 项目结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI应用入口 (包含所有API端点)
│   ├── streaming.py         # 流式传输支持
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py       # API请求模型 (包含新的trace、chunk、ws消息)
│   │   └── response.py      # API响应模型 (包含MoE、稀疏、分块响应)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── tokenizer.py     # 模拟分词器
│   │   ├── embedding.py     # 嵌入层模拟
│   │   ├── transformer.py   # Transformer层模拟 (集成MoE和稀疏注意力)
│   │   ├── moe.py          # MoE (Mixture of Experts) 实现
│   │   └── sparse_attention.py  # 稀疏注意力实现
│   └── utils/
│       ├── __init__.py
│       └── config.py        # 模型配置参数 (包含MoE、稀疏、可视化配置)
├── requirements.txt
└── README.md
```

## 安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 运行

### 开发模式（带热重载）

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产模式

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

服务启动后，访问：
- API服务：http://localhost:8000
- 交互式文档：http://localhost:8000/docs
- ReDoc文档：http://localhost:8000/redoc

## API端点

### 1. 初始化会话 (扩展版)

**POST** `/api/init`

创建一个新的计算会话，支持MoE、稀疏注意力和可视化配置。

**请求体：**
```json
{
  "text": "hello world",
  "config": {
    "n_vocab": 50257,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "d_k": 64,
    "max_seq_len": 512,
    "attention_type": "sparse",
    "sparse_config": {
      "pattern": "sliding_window",
      "window_size": 3,
      "block_size": 4
    },
    "moe_config": {
      "enabled": true,
      "n_experts": 8,
      "top_k": 2
    },
    "viz_config": {
      "render_mode": "3d",
      "animate_transitions": true,
      "precision": 4,
      "chunk_size": 100
    }
  }
}
```

**响应体：**
```json
{
  "session_id": "uuid",
  "tokens": [0, 1],
  "token_texts": ["hello", "world"],
  "total_steps": 150,
  "initial_state": {
    "embeddings": [[...], [...]],
    "positional_encodings": [[...], [...]]
  },
  "sparse_info": {
    "pattern": "sliding_window",
    "sparsity_ratio": 0.75,
    "total_elements": 16,
    "zero_elements": 12
  },
  "moe_info": {
    "enabled": true,
    "n_experts": 8,
    "top_k": 2
  }
}
```

### 2. 获取计算步骤 (扩展版)

**POST** `/api/step`

获取指定步骤的详细计算信息，包含稀疏和MoE信息。

**请求体：**
```json
{
  "session_id": "uuid",
  "step": 0
}
```

**响应体：**
```json
{
  "step": 0,
  "step_type": "layer_norm_1",
  "layer_index": 0,
  "description": "对输入进行Layer Normalization（注意力前）",
  "input_data": [[...]],
  "output_data": [[...]],
  "metadata": {
    "mean": 0.0,
    "variance": 1.0,
    "eps": 1e-5
  },
  "sparsity_info": {
    "pattern": "sliding_window",
    "sparsity_ratio": 0.75,
    "mask_matrix": [[...]]
  },
  "moe_routing": {
    "top_k_experts": [
      {
        "expert_id": 2,
        "gate_probability": 0.7,
        "output": [[...]]
      }
    ],
    "gate_logits": [0.1, 0.7, 0.2],
    "combined_output": [[...]]
  }
}
```

### 3. 获取完整计算轨迹

**POST** `/api/trace`

获取指定输入的一次性全量计算trace（用于滚动叙事与离线演示）。

**请求体：**
```json
{
  "session_id": "uuid",
  "include_metadata": true,
  "precision": 4
}
```

**响应体：**
```json
{
  "session_id": "uuid",
  "input_text": "hello world",
  "tokens": [0, 1],
  "token_texts": ["hello", "world"],
  "total_steps": 150,
  "computation_trace": [...],
  "final_output": [[...]],
  "execution_time_ms": 125.5
}
```

### 4. 分块数据获取

**POST** `/api/chunk`

获取分块数据用于分页加载，支持大负载处理。

**请求体：**
```json
{
  "session_id": "uuid",
  "chunk_size": 100,
  "chunk_id": 0
}
```

**响应体：**
```json
{
  "chunk_id": 0,
  "total_chunks": 2,
  "data": [...],
  "has_more": true
}
```

### 5. 流式传输 (SSE)

**GET** `/api/stream/{session_id}`

使用Server-Sent Events进行步骤级流式推送。

**响应：** 流式JSON数据
```json
data: {"type": "start", "data": {"total_steps": 150}}

data: {"type": "step", "data": {"step": 0, "step_type": "layer_norm_1", ...}}

data: {"type": "complete", "data": {"message": "Streaming completed"}}
```

### 6. WebSocket实时通信

**WebSocket** `/ws/{session_id}`

支持双向实时通信，可订阅、请求特定步骤或完整轨迹。

**消息格式：**
```json
// 订阅流式传输
{"type": "subscribe", "session_id": "uuid"}

// 请求特定步骤
{"type": "step", "data": {"step": 5}}

// 请求完整轨迹
{"type": "trace", "session_id": "uuid"}

// 心跳检测
{"type": "ping"}
```

### 7. 会话管理

**DELETE** `/api/session/{session_id}`
删除指定的会话。

**GET** `/api/sessions`
列出当前所有活跃的会话。

### 8. 健康检查和缓存管理

**GET** `/health`
基础健康检查。

**GET** `/api/health`
详细健康检查（包含性能指标）。

**GET** `/api/cache/stats`
获取缓存统计信息。

**DELETE** `/api/cache/clear`
清空所有缓存。

## 计算步骤类型

每个Transformer层包含以下计算步骤：

### 标准Transformer步骤
1. **layer_norm_1** - 注意力前的Layer Normalization
2. **q_projection** - Query投影
3. **k_projection** - Key投影
4. **v_projection** - Value投影
5. **attention_scores** - 注意力分数计算
6. **attention_weights** - Softmax归一化
7. **attention_output** - 注意力输出
8. **attention_projection** - 输出投影
9. **residual_1** - 第一个残差连接
10. **layer_norm_2** - FFN前的Layer Normalization
11. **ffn_hidden** - FFN第一层
12. **ffn_relu** - ReLU激活
13. **ffn_output** - FFN第二层
14. **residual_2** - 第二个残差连接

### 稀疏注意力步骤 (当启用时)
15. **sparse_mask** - 生成稀疏注意力掩码
16. **masked_attention** - 应用稀疏掩码到注意力权重
17. **sparse_attention_output** - 稀疏注意力输出

### MoE步骤 (当启用时)
18. **moe_gate_logits** - 计算MoE门控logits
19. **moe_gate_probs** - MoE门控概率
20. **expert_N_output** - 专家N输出 (N=0-7)
21. **moe_combined_output** - MoE专家组合输出

## 配置参数说明

### 基础配置
- `n_vocab`: 词汇表大小（默认：50257）
- `n_embd`: 嵌入维度（默认：768）
- `n_layer`: Transformer层数（默认：12）
- `n_head`: 多头注意力头数（默认：12）
- `d_k`: 每个注意力头的维度（默认：64，应等于n_embd/n_head）
- `max_seq_len`: 最大序列长度（默认：512）
- `attention_type`: 注意力类型（"standard" | "sparse"）

### 稀疏注意力配置 (`sparse_config`)
- `pattern`: 稀疏模式
  - `"dense"` - 稠密注意力
  - `"sliding_window"` - 滑动窗口
  - `"global_local"` - 全局+局部
  - `"blocked"` - 分块
  - `"random"` - 随机
  - `"custom"` - 自定义
- `window_size`: 滑动窗口大小（默认：3）
- `block_size`: 分块大小（默认：4）
- `global_tokens`: 全局token索引列表
- `random_ratio`: 随机稀疏比例（默认：0.1）
- `custom_mask`: 自定义掩码矩阵

### MoE配置 (`moe_config`)
- `enabled`: 是否启用MoE（默认：false）
- `n_experts`: 专家数量（默认：8）
- `top_k`: 选择的top-k专家数量（默认：2）
- `gate_noise`: 门控噪声（默认：0.0）
- `gate_dropout`: 门控dropout率（默认：0.0）

### 可视化配置 (`viz_config`)
- `render_mode`: 渲染模式（"2d" | "3d" | "graph" | "timeline"）
- `animate_transitions`: 是否启用过渡动画（默认：true）
- `show_metadata`: 是否显示元数据（默认：true）
- `precision`: 浮点数精度（1-8，默认：4）
- `chunk_size`: 数据分块大小（可选）
- `enable_compression`: 是否启用压缩（默认：true）

## 示例

### 使用curl

#### 基础示例
```bash
# 初始化会话
curl -X POST "http://localhost:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "config": {
      "n_vocab": 50257,
      "n_embd": 768,
      "n_layer": 2,
      "n_head": 12,
      "d_k": 64,
      "max_seq_len": 512
    }
  }'

# 获取步骤信息
curl -X POST "http://localhost:8000/api/step" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "step": 0
  }'
```

#### 高级示例 (启用MoE和稀疏注意力)
```bash
# 初始化会话 (MoE + 稀疏注意力)
curl -X POST "http://localhost:8000/api/init" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "config": {
      "n_vocab": 50257,
      "n_embd": 768,
      "n_layer": 2,
      "n_head": 12,
      "d_k": 64,
      "max_seq_len": 512,
      "attention_type": "sparse",
      "sparse_config": {
        "pattern": "sliding_window",
        "window_size": 3
      },
      "moe_config": {
        "enabled": true,
        "n_experts": 4,
        "top_k": 2
      },
      "viz_config": {
        "render_mode": "3d",
        "precision": 4
      }
    }
  }'

# 获取完整轨迹
curl -X POST "http://localhost:8000/api/trace" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "include_metadata": true,
    "precision": 4
  }'

# 获取分块数据
curl -X POST "http://localhost:8000/api/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "chunk_size": 50,
    "chunk_id": 0
  }'
```

#### 流式传输示例
```bash
# SSE流式传输
curl -N "http://localhost:8000/api/stream/your-session-id"
```

### 使用Python

#### 基础Python示例
```python
import requests

# 初始化会话
response = requests.post("http://localhost:8000/api/init", json={
    "text": "hello world",
    "config": {
        "n_vocab": 50257,
        "n_embd": 768,
        "n_layer": 2,
        "n_head": 12,
        "d_k": 64,
        "max_seq_len": 512
    }
})

data = response.json()
session_id = data["session_id"]
total_steps = data["total_steps"]

# 遍历所有步骤
for step in range(total_steps):
    step_response = requests.post("http://localhost:8000/api/step", json={
        "session_id": session_id,
        "step": step
    })
    step_data = step_response.json()
    print(f"Step {step}: {step_data['step_type']} - {step_data['description']}")
```

#### 高级Python示例 (MoE + 稀疏注意力)
```python
import requests
import json

# 初始化会话 (启用所有高级功能)
response = requests.post("http://localhost:8000/api/init", json={
    "text": "hello world transformer",
    "config": {
        "n_vocab": 50257,
        "n_embd": 768,
        "n_layer": 2,
        "n_head": 12,
        "d_k": 64,
        "max_seq_len": 512,
        "attention_type": "sparse",
        "sparse_config": {
            "pattern": "global_local",
            "window_size": 3,
            "global_tokens": [0, 2]  # 第一个和第三个token作为全局token
        },
        "moe_config": {
            "enabled": True,
            "n_experts": 8,
            "top_k": 2,
            "gate_noise": 0.1
        },
        "viz_config": {
            "render_mode": "timeline",
            "animate_transitions": True,
            "precision": 4,
            "chunk_size": 50
        }
    }
})

data = response.json()
session_id = data["session_id"]
print(f"稀疏信息: {data.get('sparse_info')}")
print(f"MoE信息: {data.get('moe_info')}")

# 获取完整轨迹
trace_response = requests.post("http://localhost:8000/api/trace", json={
    "session_id": session_id,
    "include_metadata": True,
    "precision": 4
})
trace_data = trace_response.json()
print(f"执行时间: {trace_data['execution_time_ms']:.2f}ms")
print(f"总步骤数: {trace_data['total_steps']}")

# 分块获取数据
chunk_id = 0
while True:
    chunk_response = requests.post("http://localhost:8000/api/chunk", json={
        "session_id": session_id,
        "chunk_size": 20,
        "chunk_id": chunk_id
    })
    chunk_data = chunk_response.json()
    
    print(f"分块 {chunk_data['chunk_id']}/{chunk_data['total_chunks']}")
    
    # 处理分块数据
    for step_data in chunk_data['data']:
        if 'moe_routing' in step_data:
            moe_info = step_data['moe_routing']
            print(f"  MoE路由: 专家 {[e['expert_id'] for e in moe_info['top_k_experts']]} 被选中")
        
        if 'sparsity_info' in step_data:
            sparse_info = step_data['sparsity_info']
            print(f"  稀疏度: {sparse_info['sparsity_ratio']:.2%}")
    
    if not chunk_data['has_more']:
        break
    chunk_id += 1
```

#### WebSocket示例
```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8000/ws/your-session-id"
    async with websockets.connect(uri) as websocket:
        # 发送订阅消息
        await websocket.send(json.dumps({
            "type": "subscribe",
            "session_id": "your-session-id"
        }))
        
        # 接收流式数据
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "step":
                step_data = data["data"]
                print(f"步骤 {step_data['step']}: {step_data['step_type']}")
            elif msg_type == "complete":
                print("流式传输完成")
                break
            elif msg_type == "error":
                print(f"错误: {data['data']['message']}")
                break

# 运行WebSocket客户端
# asyncio.run(websocket_client())
```

## 开发说明

### 添加新的计算步骤

在 `app/services/transformer.py` 中修改相应的方法，并在 `app/main.py` 的 `step_descriptions` 字典中添加描述。

### 扩展稀疏注意力模式

在 `app/services/sparse_attention.py` 中的 `SparseAttentionMask` 类中添加新的模式生成方法。

### 扩展MoE实现

在 `app/services/moe.py` 中的 `Expert` 或 `MoELayer` 类中添加新的专家类型或路由策略。

### 修改模型配置

在 `app/utils/config.py` 中修改相应的配置类（`ModelConfig`、`SparseConfig`、`MoEConfig`、`VisualizationConfig`）。

### 自定义分词器

在 `app/services/tokenizer.py` 中实现新的分词逻辑。

### 添加新的API端点

在 `app/main.py` 中添加新的路由处理函数，并在 `app/models/` 中定义相应的请求/响应模型。

## 前端集成指南

### /viz 模式
用于实时可视化，推荐使用WebSocket连接：
```javascript
// 连接WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

// 订阅流式数据
ws.send(JSON.stringify({
    type: "subscribe",
    session_id: sessionId
}));

// 处理实时数据
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "step") {
        // 渲染步骤数据
        renderStep(message.data);
    }
};
```

### /story 模式
用于滚动叙事和离线演示，推荐使用trace API：
```javascript
// 获取完整轨迹
const response = await fetch("/api/trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        session_id: sessionId,
        include_metadata: true,
        precision: 4
    })
});

const traceData = await response.json();
// 使用traceData构建叙事
buildNarrative(traceData);
```

## 部署和性能优化

### Docker部署
```bash
# 构建镜像
docker build -t transformer-api ./backend

# 运行容器
docker run -p 8000:8000 transformer-api
```

### 性能调优
- 使用 `viz_config.precision` 调整浮点精度以平衡精度和性能
- 启用 `viz_config.enable_compression` 减少网络传输
- 使用 `chunk_size` 分块处理大负载
- 调整缓存大小（默认100个条目）

### 监控
- `/health` - 基础健康检查
- `/api/health` - 详细性能指标
- `/api/cache/stats` - 缓存统计

## 注意事项

- 当前实现使用内存存储会话，服务重启后会话会丢失
- 权重矩阵使用随机初始化，不是真实的预训练模型
- WebSocket连接数有限制，建议在生产环境中添加连接池管理
- 稀疏注意力和MoE会增加计算复杂度，建议根据硬件配置调整参数
- GZip压缩已启用，但客户端需要支持Accept-Encoding: gzip
- 建议在生产环境中添加会话过期、清理机制和负载均衡

## 更新日志

### v2.0.0 (当前版本)
- ✅ 添加MoE (Mixture of Experts) 支持
- ✅ 添加稀疏注意力模式（滑动窗口、全局+局部、分块、随机、自定义）
- ✅ 添加SSE和WebSocket流式传输
- ✅ 添加数据分块功能
- ✅ 添加可视化配置选项
- ✅ 添加完整trace API
- ✅ 添加可配置浮点精度
- ✅ 更新API文档和示例

### v1.0.0
- ✅ 基础Transformer计算模拟
- ✅ 标准API端点
- ✅ 会话管理
- ✅ 缓存机制
- ✅ GZip压缩

## 许可证

MIT
