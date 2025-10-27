# Transformer计算模拟器 - 后端服务

标准Transformer逐步计算可视化的FastAPI后端服务。

## 功能特性

- 🚀 基于FastAPI的高性能异步API
- 🔢 使用NumPy实现完整的Transformer计算流程
- 📊 逐步追踪和记录所有中间计算结果
- 🎯 支持多会话并发处理
- 📝 自动生成的API文档

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
│   ├── main.py              # FastAPI应用入口
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py       # API请求模型
│   │   └── response.py      # API响应模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── tokenizer.py     # 模拟分词器
│   │   ├── embedding.py     # 嵌入层模拟
│   │   └── transformer.py   # Transformer层模拟
│   └── utils/
│       ├── __init__.py
│       └── config.py        # 模型配置参数
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

### 1. 初始化会话

**POST** `/api/init`

创建一个新的计算会话，对输入文本进行分词和嵌入，并预计算所有Transformer层的中间结果。

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
    "max_seq_len": 512
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
  }
}
```

### 2. 获取计算步骤

**POST** `/api/step`

获取指定步骤的详细计算信息。

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
  }
}
```

### 3. 删除会话

**DELETE** `/api/session/{session_id}`

删除指定的会话。

### 4. 列出所有会话

**GET** `/api/sessions`

列出当前所有活跃的会话。

## 计算步骤类型

每个Transformer层包含以下计算步骤：

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

## 配置参数说明

- `n_vocab`: 词汇表大小（默认：50257）
- `n_embd`: 嵌入维度（默认：768）
- `n_layer`: Transformer层数（默认：12）
- `n_head`: 多头注意力头数（默认：12）
- `d_k`: 每个注意力头的维度（默认：64，应等于n_embd/n_head）
- `max_seq_len`: 最大序列长度（默认：512）

## 示例

### 使用curl

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

### 使用Python

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

## 开发说明

### 添加新的计算步骤

在 `app/services/transformer.py` 中修改相应的方法，并在 `app/main.py` 的 `step_descriptions` 字典中添加描述。

### 修改模型配置

在 `app/utils/config.py` 中修改 `ModelConfig` 类。

### 自定义分词器

在 `app/services/tokenizer.py` 中实现新的分词逻辑。

## 注意事项

- 当前实现使用内存存储会话，服务重启后会话会丢失
- 权重矩阵使用随机初始化，不是真实的预训练模型
- 此版本只实现标准Transformer，不包含MoE或稀疏注意力
- 建议在生产环境中添加会话过期和清理机制

## 许可证

MIT
