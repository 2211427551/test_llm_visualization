# API端点实现说明

本文档描述了基于FastAPI的模型API端点实现。

## 实现的功能

### 1. `/initialize` GET 接口

**功能描述：**
- 加载Transformer模型
- 返回模型基本配置信息
- 支持自定义配置参数

**请求参数：**
- `config` (可选): JSON字符串，包含模型配置参数

**返回结构：**
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

### 2. `/forward` POST 接口

**功能描述：**
- 接收文本输入（batch size=1）
- 完成分词、张量化、模型前向传播
- 调用数据捕获器（可选）
- 返回推理结果

**请求结构：**
```json
{
  "text": "Hello world",
  "capture_data": false
}
```

**返回结构：**
```json
{
  "success": true,
  "message": "前向传播完成",
  "logits_shape": [1, 5, 1000],
  "sequence_length": 5,
  "captured_data": null
}
```

### 3. `/status` GET 接口

**功能描述：**
- 获取模型当前状态
- 检查模型是否已初始化

**返回结构：**
```json
{
  "initialized": true,
  "device": "cpu",
  "vocab_size": 50304,
  "n_layer": 12,
  ...
}
```

## 异常处理

### 中文错误描述

所有异常情况都返回带中文错误描述的HTTP状态码：

1. **模型未初始化** (HTTP 400)
   ```json
   {
     "detail": "模型未初始化，请先调用 /initialize 接口"
   }
   ```

2. **输入为空** (HTTP 400)
   ```json
   {
     "detail": "输入文本不能为空"
   }
   ```

3. **序列长度超限** (HTTP 400)
   ```json
   {
     "detail": "输入序列长度 X 超过了模型最大长度 Y"
   }
   ```

4. **模型推理错误** (HTTP 500)
   ```json
   {
     "detail": "模型推理错误: 具体错误信息"
   }
   ```

5. **服务器内部错误** (HTTP 500)
   ```json
   {
     "detail": "服务器内部错误: 具体错误信息"
   }
   ```

## 核心组件

### 1. 分词器服务 (`app/services/tokenizer.py`)

- 实现简单的字符级和词级分词
- 支持中英文混合文本
- 提供BOS、EOS、UNK、PAD等特殊token
- 可配置词表大小

### 2. 模型服务 (`app/services/model.py`)

- 单例模式管理模型实例
- 线程安全的模型加载和推理
- 支持GPU/CPU设备选择
- 集成数据捕获功能

### 3. API模式 (`app/schemas/api.py`)

- 使用Pydantic进行数据验证
- 完整的请求/响应模式定义
- 自动类型检查和转换

### 4. 路由 (`app/routers/model.py`)

- RESTful API设计
- 完整的错误处理
- 详细的API文档

## 数据捕获功能

支持完整的前向传播数据捕获：

- 嵌入层数据
- 注意力权重信息
- MoE路由信息（如果启用）
- 中间张量状态
- 性能统计信息

## 测试

### 集成测试 (`tests/test_model_api.py`)

使用TestClient进行全面的API测试：

1. **初始化测试**
   - 默认配置初始化
   - 自定义配置初始化
   - 重复初始化处理

2. **前向传播测试**
   - 英文文本处理
   - 中文文本处理
   - 数据捕获功能
   - 空文本异常处理

3. **异常处理测试**
   - 未初始化模型访问
   - 无效请求格式
   - 文本长度验证

4. **响应格式验证**
   - 字段完整性检查
   - 数据类型验证
   - 边界条件测试

### 快速验证脚本 (`validate_implementation.py`)

验证所有组件的正确性：
- 模块导入测试
- 分词器功能测试
- 模型服务测试
- 前向推理测试
- 数据捕获测试
- API模式测试

## 使用示例

### 1. 初始化模型

```bash
curl -X GET "http://localhost:8000/api/v1/initialize"
```

### 2. 基础前向传播

```bash
curl -X POST "http://localhost:8000/api/v1/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "capture_data": false
  }'
```

### 3. 带数据捕获的前向传播

```bash
curl -X POST "http://localhost:8000/api/v1/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "capture_data": true
  }'
```

### 4. 自定义配置初始化

```bash
curl -X GET "http://localhost:8000/api/v1/initialize?config={\"n_layer\":6,\"n_head\":8,\"use_moe\":true}"
```

## 技术特点

1. **模块化设计**：清晰的代码结构，易于维护和扩展
2. **类型安全**：使用Pydantic进行严格的数据验证
3. **错误处理**：完善的异常处理机制，中文错误信息
4. **性能优化**：支持GPU加速，内存管理优化
5. **可扩展性**：预留稀疏注意力和MoE功能接口
6. **测试覆盖**：全面的单元测试和集成测试

## 依赖项

- FastAPI: Web框架
- PyTorch: 深度学习框架
- Pydantic: 数据验证
- Uvicorn: ASGI服务器
- httpx: HTTP客户端（测试用）

## 启动服务

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API文档地址：`http://localhost:8000/docs`