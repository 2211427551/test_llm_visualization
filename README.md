# Transformer计算模拟器

一个用于可视化和学习Transformer模型内部计算过程的教育工具。本项目提供了标准Transformer架构的逐步计算模拟，帮助理解注意力机制、前馈网络和残差连接等关键概念。

> 📖 **[中文文档](README_zh.md)** | **[English Documentation](README.md)**

## 项目概述

本项目实现了一个完整的Transformer计算模拟器，包括：

- **后端服务**：基于FastAPI的RESTful API，提供Transformer计算模拟
- **前端应用**：基于Next.js 14的现代化Web界面，提供交互式可视化（D3.js）
- **逐步可视化**：记录并返回每个计算步骤的中间结果
- **高级特性**：稀疏注意力（Sparse Attention）、混合专家模型（MoE）
- **完整输出层**：Logits Head、Softmax、预测结果可视化
- **教育友好**：清晰的步骤描述和详细的元数据

## 功能特性

### 核心功能
- ✅ 标准Transformer架构实现
- ✅ 多头自注意力机制（Multi-Head Attention）
- ✅ 前馈神经网络（FFN）
- ✅ Layer Normalization
- ✅ 残差连接（Residual Connections）
- ✅ 位置编码（Positional Encoding）
- ✅ 完整的计算步骤追踪
- ✅ 会话管理
- ✅ CORS支持

### 高级特性
- ✅ **稀疏注意力**：Sliding Window、Global-Local、Blocked、Random等多种模式
- ✅ **混合专家模型（MoE）**：智能路由、Top-K选择、负载均衡可视化
- ✅ **输出层可视化**：Logits Head、Softmax、Top-K预测、概率分布

### 可视化功能（D3.js）
- ✅ Token化与嵌入可视化
- ✅ 注意力权重热图
- ✅ 稀疏注意力掩码
- ✅ MoE路由决策
- ✅ 矩阵运算动画
- ✅ 概率分布图表
- ✅ 实时交互与悬停提示

## 技术栈

### 后端
- Python 3.9+
- FastAPI - 现代化的Web框架
- NumPy - 高性能矩阵计算
- Pydantic - 数据验证和设置管理
- Uvicorn - ASGI服务器

### 前端
- Next.js 14+ (App Router)
- TypeScript
- D3.js v7 - 数据可视化
- Tailwind CSS - UI样式
- Zustand - 状态管理
- Axios - HTTP客户端

## 快速开始

### 1. 启动后端服务

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- API文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/

### 2. 启动前端应用

在另一个终端窗口中：

```bash
cd frontend
npm install
npm run dev
```

前端应用访问：http://localhost:3000

### 3. 开始使用

1. 在前端页面输入文本（例如："Hello world"）
2. 点击"开始计算"按钮
3. 使用控制面板浏览计算步骤
4. 查看可视化结果和详细解释

### 运行测试

```bash
cd backend
source venv/bin/activate
python test_api.py
```

## API使用示例

### 1. 初始化会话

```bash
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
```

### 2. 获取计算步骤

```bash
curl -X POST "http://localhost:8000/api/step" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "step": 0
  }'
```

## 计算步骤

每个Transformer层包含14个计算步骤：

1. **layer_norm_1** - 注意力前的Layer Normalization
2. **q_projection** - Query矩阵投影
3. **k_projection** - Key矩阵投影
4. **v_projection** - Value矩阵投影
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

## 项目结构

```
.
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── main.py         # FastAPI应用入口
│   │   ├── models/         # Pydantic数据模型
│   │   ├── services/       # 核心业务逻辑
│   │   └── utils/          # 工具和配置
│   ├── requirements.txt    # Python依赖
│   ├── test_api.py        # API测试脚本
│   └── README.md          # 后端文档
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── app/           # Next.js应用路由
│   │   ├── components/    # React组件
│   │   ├── store/         # Zustand状态管理
│   │   ├── services/      # API调用服务
│   │   └── types/         # TypeScript类型定义
│   ├── public/            # 静态资源
│   ├── package.json       # Node.js依赖
│   ├── SETUP.md          # 前端设置指南
│   └── README.md         # 前端文档
└── README.md             # 项目主文档
```

## 配置参数

- `n_vocab`: 词汇表大小（默认：50257）
- `n_embd`: 嵌入维度（默认：768）
- `n_layer`: Transformer层数（默认：12）
- `n_head`: 多头注意力头数（默认：12）
- `d_k`: 每个注意力头的维度（默认：64）
- `max_seq_len`: 最大序列长度（默认：512）

## 开发指南

### 添加新功能

1. 在 `app/services/transformer.py` 中实现新的计算步骤
2. 在 `app/main.py` 中添加步骤描述
3. 更新 `app/models/` 中的数据模型（如需要）
4. 编写测试用例

### 运行开发服务器

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 注意事项

- 权重使用随机初始化，不是预训练模型
- 会话数据存储在内存中，服务重启后会丢失
- 当前版本只实现标准Transformer，不包含MoE或稀疏注意力
- 适用于教育和学习目的，不建议用于生产环境

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 路线图

### 已完成 ✅
- [x] 添加前端可视化界面
- [x] 集成D3.js进行交互式可视化
- [x] 支持MoE（Mixture of Experts）
- [x] 支持稀疏注意力机制
- [x] 输出层完整可视化（Logits、Softmax、预测）
- [x] 完整的中文文档

### 进行中 🚧
- [ ] 预训练模型加载支持
- [ ] 更多解码策略（Beam Search、Top-P Sampling）
- [ ] 性能优化和缓存

### 计划中 📋
- [ ] 添加更多的分词器选项
- [ ] 实现会话持久化（数据库）
- [ ] 支持批处理
- [ ] 多语言界面（英文、日文等）
- [ ] 移动端适配
- [ ] 模型对比工具
