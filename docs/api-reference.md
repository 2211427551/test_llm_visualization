# API 接口参考文档

## 📖 概述

本文档详细描述了 Transformer 深度学习平台的 RESTful API 接口。所有接口都基于 FastAPI 框架构建，提供完整的 OpenAPI 文档支持。

## 🔗 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **API 版本**: v1
- **内容类型**: `application/json`
- **字符编码**: UTF-8

### 认证方式

目前 API 不需要认证，适用于开发和测试环境。生产环境建议添加适当的认证机制。

## 📋 接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/initialize` | 初始化模型 |
| POST | `/forward` | 执行前向传播 |
| GET | `/status` | 获取模型状态 |
| GET | `/health/health` | 健康检查 |
| GET | `/health/ping` | 连通性检查 |

## 🚀 接口详情

### 1. 模型初始化

#### `GET /initialize`

初始化 Transformer 模型，支持自定义配置参数。

**请求参数：**

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| config | string | 否 | null | 模型配置的 JSON 字符串 |

**请求示例：**

```bash
# 使用默认配置
curl -X GET "http://localhost:8000/api/v1/initialize"

# 使用自定义配置
curl -X GET "http://localhost:8000/api/v1/initialize?config={\"n_layer\":6,\"use_sparse_attention\":true,\"use_moe\":true}"
```

**响应结构：**

```json
{
  "success": true,
  "message": "模型初始化成功",
  "config": {
    "vocab_size": 50304,
    "context_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embed": 768,
    "dropout": 0.1,
    "bias": true,
    "ffn_hidden_multiplier": 4,
    "use_sparse_attention": false,
    "use_moe": false,
    "moe_num_experts": 8,
    "moe_top_k": 2,
    "moe_activation": "gelu",
    "moe_dropout": null,
    "device": "cpu"
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 初始化是否成功 |
| message | string | 响应消息 |
| config | object | 模型配置信息 |

**错误响应：**

```json
{
  "detail": "配置参数无效: n_embed 必须能被 n_head 整除"
}
```

### 2. 前向传播

#### `POST /forward`

对输入文本执行模型前向传播，返回推理结果。

**请求结构：**

```json
{
  "text": "深度学习是人工智能的重要分支",
  "capture_data": false,
  "max_length": null
}
```

**请求字段说明：**

| 字段 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| text | string | 是 | - | 输入文本 |
| capture_data | boolean | 否 | false | 是否捕获中间数据 |
| max_length | integer | 否 | null | 最大序列长度限制 |

**请求示例：**

```bash
# 基础推理
curl -X POST "http://localhost:8000/api/v1/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "深度学习是人工智能的重要分支",
    "capture_data": false
  }'

# 带数据捕获的推理
curl -X POST "http://localhost:8000/api/v1/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "深度学习是人工智能的重要分支",
    "capture_data": true,
    "max_length": 512
  }'
```

**响应结构：**

```json
{
  "success": true,
  "message": "前向传播完成",
  "logits_shape": [1, 16, 50304],
  "sequence_length": 16,
  "captured_data": null,
  "processing_time": 0.234
}
```

**响应字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 推理是否成功 |
| message | string | 响应消息 |
| logits_shape | array | 输出 logits 的形状 |
| sequence_length | integer | 输入序列长度 |
| captured_data | object/null | 捕获的中间数据 |
| processing_time | number | 处理时间（秒） |

**捕获数据结构（当 capture_data=true 时）：**

```json
{
  "captured_data": {
    "embeddings": {
      "token_embeddings": {
        "shape": [1, 16, 768],
        "dtype": "torch.float32"
      },
      "position_embeddings": {
        "shape": [1, 16, 768],
        "dtype": "torch.float32"
      }
    },
    "layers": [
      {
        "layer_idx": 0,
        "attention": {
          "attn_weights_shape": [1, 12, 16, 16]
        },
        "moe": {
          "expert_usage": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
          "load_balance_loss": 0.001234
        }
      }
    ],
    "performance": {
      "forward_time": 0.234,
      "memory_usage": 156.7
    }
  }
}
```

**错误响应：**

```json
{
  "detail": "输入文本不能为空"
}
```

```json
{
  "detail": "输入序列长度 2048 超过了模型最大长度 1024"
}
```

### 3. 模型状态

#### `GET /status`

获取当前模型的运行状态和配置信息。

**请求示例：**

```bash
curl -X GET "http://localhost:8000/api/v1/status"
```

**响应结构：**

```json
{
  "initialized": true,
  "device": "cpu",
  "config": {
    "vocab_size": 50304,
    "context_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embed": 768,
    "use_sparse_attention": false,
    "use_moe": false
  },
  "model_info": {
    "total_parameters": 124439808,
    "model_size_mb": 474.8,
    "device_memory_mb": 0.0
  },
  "performance_stats": {
    "total_inferences": 42,
    "average_inference_time": 0.156,
    "last_inference_time": 0.234
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| initialized | boolean | 模型是否已初始化 |
| device | string | 运行设备 |
| config | object | 模型配置 |
| model_info | object | 模型信息 |
| performance_stats | object | 性能统计 |

### 4. 健康检查

#### `GET /health/health`

检查服务健康状态，包括模型状态和系统资源。

**请求示例：**

```bash
curl -X GET "http://localhost:8000/api/v1/health/health"
```

**响应结构：**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "checks": {
    "model": {
      "status": "healthy",
      "initialized": true,
      "device": "cpu"
    },
    "memory": {
      "status": "healthy",
      "used_mb": 1024.5,
      "available_mb": 7168.0,
      "usage_percent": 12.5
    },
    "disk": {
      "status": "healthy",
      "used_gb": 45.2,
      "available_gb": 234.8,
      "usage_percent": 16.1
    }
  }
}
```

#### `GET /health/ping`

简单的连通性检查，返回 PONG 响应。

**请求示例：**

```bash
curl -X GET "http://localhost:8000/api/v1/health/ping"
```

**响应结构：**

```json
{
  "message": "PONG",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 数据模式

### 请求模式

#### ForwardRequest

```python
class ForwardRequest(BaseModel):
    """前向传播请求模式"""
    text: str = Field(..., min_length=1, max_length=10000, description="输入文本")
    capture_data: bool = Field(default=False, description="是否捕获中间数据")
    max_length: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=4096, 
        description="最大序列长度限制"
    )
```

#### InitializeRequest

```python
class InitializeRequest(BaseModel):
    """模型初始化请求模式"""
    config: Optional[str] = Field(
        default=None,
        description="模型配置的JSON字符串"
    )
```

### 响应模式

#### ForwardResponse

```python
class ForwardResponse(BaseModel):
    """前向传播响应模式"""
    success: bool = Field(..., description="推理是否成功")
    message: str = Field(..., description="响应消息")
    logits_shape: List[int] = Field(..., description="输出logits的形状")
    sequence_length: int = Field(..., ge=0, description="输入序列长度")
    captured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="捕获的中间数据"
    )
    processing_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="处理时间（秒）"
    )
```

#### InitializeResponse

```python
class InitializeResponse(BaseModel):
    """模型初始化响应模式"""
    success: bool = Field(..., description="初始化是否成功")
    message: str = Field(..., description="响应消息")
    config: Dict[str, Any] = Field(..., description="模型配置信息")
```

## ⚠️ 错误处理

### HTTP 状态码

| 状态码 | 描述 | 示例场景 |
|--------|------|----------|
| 200 | 请求成功 | 正常的 API 调用 |
| 400 | 请求参数错误 | 输入文本为空、配置无效 |
| 404 | 资源不存在 | API 路径错误 |
| 500 | 服务器内部错误 | 模型推理失败、内存不足 |

### 错误响应格式

```json
{
  "detail": "错误描述信息",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 常见错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| MODEL_NOT_INITIALIZED | 模型未初始化 | 先调用 `/initialize` 接口 |
| INVALID_INPUT | 输入参数无效 | 检查请求参数格式和值 |
| SEQUENCE_TOO_LONG | 序列长度超限 | 减少输入文本长度或调整模型配置 |
| INFERENCE_ERROR | 推理错误 | 检查模型状态和系统资源 |
| CONFIG_ERROR | 配置错误 | 验证配置参数的正确性 |

## 📊 使用示例

### Python 客户端

```python
import requests
import json

class TransformerClient:
    """Transformer API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def initialize_model(self, config: dict = None) -> dict:
        """初始化模型"""
        url = f"{self.base_url}/initialize"
        params = {}
        if config:
            params['config'] = json.dumps(config)
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def forward(self, text: str, capture_data: bool = False) -> dict:
        """执行前向传播"""
        url = f"{self.base_url}/forward"
        data = {
            "text": text,
            "capture_data": capture_data
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> dict:
        """获取模型状态"""
        url = f"{self.base_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# 使用示例
client = TransformerClient()

# 初始化模型
config = {
    "n_layer": 6,
    "use_sparse_attention": True,
    "use_moe": True
}
init_result = client.initialize_model(config)
print("初始化结果:", init_result)

# 执行推理
result = client.forward("深度学习是人工智能的重要分支", capture_data=True)
print("推理结果:", result)

# 获取状态
status = client.get_status()
print("模型状态:", status)
```

### JavaScript 客户端

```javascript
class TransformerClient {
    constructor(baseUrl = 'http://localhost:8000/api/v1') {
        this.baseUrl = baseUrl;
    }
    
    async initializeModel(config = null) {
        const url = new URL(`${this.baseUrl}/initialize`);
        if (config) {
            url.searchParams.append('config', JSON.stringify(config));
        }
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async forward(text, captureData = false) {
        const response = await fetch(`${this.baseUrl}/forward`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                capture_data: captureData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// 使用示例
const client = new TransformerClient();

async function example() {
    try {
        // 初始化模型
        const config = {
            n_layer: 6,
            use_sparse_attention: true,
            use_moe: true
        };
        const initResult = await client.initializeModel(config);
        console.log('初始化结果:', initResult);
        
        // 执行推理
        const result = await client.forward('深度学习是人工智能的重要分支', true);
        console.log('推理结果:', result);
        
        // 获取状态
        const status = await client.getStatus();
        console.log('模型状态:', status);
        
    } catch (error) {
        console.error('API 调用失败:', error);
    }
}

example();
```

### cURL 批处理脚本

```bash
#!/bin/bash

# API 基础 URL
BASE_URL="http://localhost:8000/api/v1"

# 初始化模型
echo "初始化模型..."
curl -s -X GET "${BASE_URL}/initialize?config={\"n_layer\":6,\"use_sparse_attention\":true}" | jq .

# 批量推理
echo "执行批量推理..."
texts=(
    "深度学习是人工智能的重要分支"
    "自然语言处理技术发展迅速"
    "Transformer架构改变了NLP领域"
)

for text in "${texts[@]}"; do
    echo "处理文本: $text"
    curl -s -X POST "${BASE_URL}/forward" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$text\",\"capture_data\":true}" | jq '.success, .sequence_length, .processing_time'
    echo "---"
done

# 获取最终状态
echo "获取模型状态..."
curl -s -X GET "${BASE_URL}/status" | jq .
```

## 🔄 WebSocket 支持

虽然当前版本主要使用 REST API，但未来版本计划支持 WebSocket 以实现实时流式推理。

### 计划中的 WebSocket 接口

```javascript
// WebSocket 连接示例
const ws = new WebSocket('ws://localhost:8000/ws/inference');

ws.onopen = function(event) {
    console.log('WebSocket 连接已建立');
    
    // 发送推理请求
    ws.send(JSON.stringify({
        type: 'inference',
        text: '深度学习是人工智能的重要分支',
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'token':
            console.log('生成的 token:', data.token);
            break;
        case 'complete':
            console.log('推理完成:', data.result);
            break;
        case 'error':
            console.error('推理错误:', data.error);
            break;
    }
};
```

## 📈 性能优化建议

### 1. 批量处理

```python
# 批量推理示例
def batch_forward(client, texts, batch_size=8):
    """批量处理多个文本"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = []
        
        for text in batch:
            result = client.forward(text, capture_data=False)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
```

### 2. 连接池

```python
import requests.adapters
import urllib3

# 配置连接池
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# 使用连接池
response = session.get(f"{base_url}/status")
```

### 3. 异步客户端

```python
import aiohttp
import asyncio

async def async_forward(session, text):
    """异步推理"""
    data = {"text": text, "capture_data": False}
    async with session.post("/api/v1/forward", json=data) as response:
        return await response.json()

async def batch_async_forward(texts):
    """异步批量推理"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_forward(session, text) for text in texts]
        return await asyncio.gather(*tasks)
```

## 📝 最佳实践

### 1. 错误处理

```python
def safe_forward(client, text, max_retries=3):
    """安全的推理调用"""
    for attempt in range(max_retries):
        try:
            result = client.forward(text)
            return result
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # 指数退避
```

### 2. 资源管理

```python
class ModelManager:
    """模型资源管理器"""
    
    def __init__(self):
        self.client = TransformerClient()
        self.initialized = False
    
    def __enter__(self):
        if not self.initialized:
            self.client.initialize_model()
            self.initialized = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源（如果需要）
        pass
    
    def infer(self, text):
        if not self.initialized:
            raise RuntimeError("模型未初始化")
        return self.client.forward(text)

# 使用上下文管理器
with ModelManager() as manager:
    result1 = manager.infer("文本1")
    result2 = manager.infer("文本2")
```

### 3. 监控和日志

```python
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_forward(client, text):
    """带监控的推理"""
    start_time = time.time()
    
    try:
        result = client.forward(text)
        processing_time = time.time() - start_time
        
        logger.info(f"推理成功: 文本长度={len(text)}, 处理时间={processing_time:.3f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"推理失败: 错误={str(e)}, 处理时间={processing_time:.3f}s")
        raise
```

---

💡 **提示**：在生产环境中使用时，建议添加适当的认证、限流和监控机制以确保服务稳定性。