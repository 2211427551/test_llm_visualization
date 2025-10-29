# Transformer可视化教育平台

一个功能完整的交互式Transformer模型可视化教育工具，帮助深入理解现代大语言模型的内部工作原理。

## 🎯 项目简介

本项目提供了一个完整的Transformer架构可视化系统，通过直观的动画和交互式界面，展示从输入到输出的每一个计算步骤。特别适合：

- 🎓 **学生和研究者**：深入理解Transformer的工作原理
- 👨‍💻 **AI工程师**：调试和优化模型架构
- 📚 **教育工作者**：辅助教学，提升课堂互动
- 🔬 **爱好者**：探索大语言模型的奥秘

## ✨ 核心特性

### 1. 完整的Transformer可视化

- ✅ **Token化与嵌入**：词元分割、词嵌入、位置编码
- ✅ **多头自注意力机制**：Query/Key/Value投影、注意力权重计算、多头融合
- ✅ **前馈神经网络（FFN）**：两层全连接网络、ReLU激活
- ✅ **Layer Normalization**：归一化处理，稳定训练
- ✅ **残差连接**：信息保留，梯度流动
- ✅ **输出层**：Logits Head、Softmax、预测结果

### 2. 高级特性支持

#### 稀疏注意力机制
- 🔹 **Dense（标准注意力）**：全连接注意力模式
- 🔹 **Sliding Window**：滑动窗口，降低计算复杂度
- 🔹 **Global-Local**：全局token + 局部注意力
- 🔹 **Blocked Attention**：分块注意力模式
- 🔹 **Random Attention**：随机稀疏连接
- 🔹 **Custom Pattern**：自定义注意力掩码

#### Mixture of Experts (MoE)
- 🔸 **智能路由**：动态选择专家网络
- 🔸 **Top-K选择**：每个token选择K个专家
- 🔸 **负载均衡**：可视化专家利用率
- 🔸 **门控机制**：Softmax路由权重计算

### 3. 交互式可视化

- 🎨 **D3.js渲染**：流畅的SVG动画效果
- 🖱️ **实时交互**：悬停查看详细信息
- 🎮 **动画控制**：播放、暂停、步进、重置
- 🔍 **数据探索**：矩阵热图、向量可视化、概率分布

### 4. 教育友好设计

- 📖 **同步解释面板**：每个步骤的详细文字说明
- 📊 **数据形状展示**：清晰标注张量维度
- 🎯 **关键概念高亮**：突出显示重要计算步骤
- 🌐 **中文界面**：完整的中文文档和UI

## 🛠️ 技术栈

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.9+ | 主要编程语言 |
| **FastAPI** | 最新版 | 高性能Web框架 |
| **NumPy** | 最新版 | 高效矩阵运算 |
| **Pydantic** | v2 | 数据验证和序列化 |
| **Uvicorn** | 最新版 | ASGI服务器 |

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **Next.js** | 14+ | React框架（App Router） |
| **TypeScript** | 5+ | 类型安全 |
| **D3.js** | v7 | 数据可视化 |
| **Tailwind CSS** | 3+ | UI样式 |
| **Zustand** | 最新版 | 轻量级状态管理 |
| **Axios** | 最新版 | HTTP客户端 |

## 🚀 快速开始

### 方式一：Docker 部署（推荐）

使用 Docker Compose 一键启动所有服务：

```bash
# 构建并启动
docker compose up --build

# 或在后台运行
docker compose up -d --build
```

服务启动后访问：
- 🌐 **前端应用**: http://localhost:3000
- 📚 **API文档**: http://localhost:8000/docs
- ❤️ **健康检查**: http://localhost:8000/health

> 📖 详细的 Docker 部署指南请查看 [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
>
> 🔧 遇到问题？运行诊断脚本：`./diagnose-docker.sh`

### 方式二：本地开发

#### 环境要求

- **Node.js**: 18.0+ 
- **Python**: 3.9+
- **npm** 或 **yarn**
- **pip**: Python包管理器

#### 1. 克隆项目

```bash
git clone <repository-url>
cd transformer-visualization
```

#### 2. 后端部署

##### 步骤1：创建虚拟环境

```bash
cd backend

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

##### 步骤2：安装依赖

```bash
pip install -r requirements.txt
```

##### 步骤3：启动后端服务

```bash
# 开发模式（自动重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

##### 验证后端

访问以下地址确认服务正常运行：

- 🏠 **首页**: http://localhost:8000/
- 📚 **API文档**: http://localhost:8000/docs
- 📖 **ReDoc文档**: http://localhost:8000/redoc
- ❤️ **健康检查**: http://localhost:8000/health

#### 3. 前端部署

在新的终端窗口中：

##### 步骤1：安装依赖

```bash
cd frontend
npm install
# 或使用 yarn
yarn install
```

##### 步骤2：启动开发服务器

```bash
npm run dev
# 或
yarn dev
```

##### 步骤3：访问应用

打开浏览器访问：**http://localhost:3000**

### 4. 生产环境部署

#### 前端构建

```bash
cd frontend
npm run build
npm start
```

#### 后端部署（使用Gunicorn）

```bash
cd backend
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### 使用Docker（可选）

```bash
# 构建镜像
docker build -t transformer-viz-backend ./backend
docker build -t transformer-viz-frontend ./frontend

# 运行容器
docker run -d -p 8000:8000 transformer-viz-backend
docker run -d -p 3000:3000 transformer-viz-frontend
```

## 📖 使用指南

### 基本使用流程

1. **输入文本**
   - 在输入框中输入要处理的文本（例如："Hello world"）
   - 点击"开始计算"按钮

2. **观察可视化**
   - 系统会初始化会话并返回token信息
   - 可视化画布会显示当前步骤的计算过程

3. **控制动画**
   - **播放/暂停**: 自动播放动画或暂停
   - **下一步**: 手动前进到下一个计算步骤
   - **上一步**: 返回上一个计算步骤
   - **重置**: 回到初始状态

4. **探索细节**
   - 将鼠标悬停在可视化元素上查看详细数据
   - 阅读右侧解释面板了解当前步骤的原理
   - 查看数据形状和层信息

### 高级功能

#### 切换稀疏注意力模式

```typescript
// 在配置中指定稀疏模式
const config = {
  n_vocab: 50257,
  n_embd: 768,
  n_layer: 12,
  n_head: 12,
  d_k: 64,
  max_seq_len: 512,
  attention_type: 'sparse',  // 启用稀疏注意力
  sparse_config: {
    pattern: 'sliding_window',
    window_size: 256
  }
};
```

支持的稀疏模式：
- `dense`: 标准全注意力
- `sliding_window`: 滑动窗口（指定window_size）
- `global_local`: 全局+局部（指定global_tokens）
- `blocked`: 分块注意力（指定block_size）
- `random`: 随机稀疏（指定random_ratio）
- `custom`: 自定义掩码（提供custom_mask）

#### 观察MoE路由

在MoE层可视化中，可以看到：
- 每个token的专家选择
- 路由权重分布
- 专家负载均衡情况
- 门控网络输出

## 📂 项目结构

```
transformer-visualization/
│
├── backend/                      # 后端服务
│   ├── app/
│   │   ├── main.py              # FastAPI应用入口
│   │   ├── models/              # Pydantic数据模型
│   │   │   ├── request.py       # 请求模型
│   │   │   └── response.py      # 响应模型
│   │   ├── services/            # 核心业务逻辑
│   │   │   ├── tokenizer.py    # 分词器
│   │   │   ├── embedding.py    # 嵌入层
│   │   │   ├── transformer.py  # Transformer模拟器
│   │   │   ├── attention.py    # 注意力机制
│   │   │   └── moe.py          # MoE实现
│   │   └── utils/              # 工具函数
│   │       └── config.py       # 配置管理
│   ├── requirements.txt        # Python依赖
│   ├── test_api.py            # API测试脚本
│   └── README.md              # 后端文档
│
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── app/               # Next.js App Router
│   │   │   ├── page.tsx       # 主页
│   │   │   ├── layout.tsx     # 布局
│   │   │   └── globals.css    # 全局样式
│   │   ├── components/        # React组件
│   │   │   ├── ControlPanel.tsx        # 控制面板
│   │   │   ├── ExplanationPanel.tsx    # 解释面板
│   │   │   ├── InputModule.tsx         # 输入模块
│   │   │   ├── VisualizationCanvas.tsx # 可视化画布
│   │   │   └── visualizations/         # 可视化组件
│   │   │       ├── TokenizationViz.tsx       # 分词可视化
│   │   │       ├── EmbeddingViz.tsx          # 嵌入可视化
│   │   │       ├── MultiHeadAttentionViz.tsx # 多头注意力
│   │   │       ├── SparseAttentionViz.tsx    # 稀疏注意力
│   │   │       ├── MoEFFNViz.tsx             # MoE前馈网络
│   │   │       ├── OutputLayerViz.tsx        # 输出层可视化
│   │   │       └── index.ts                  # 导出文件
│   │   ├── services/          # API服务
│   │   │   └── api.ts         # API客户端
│   │   ├── store/             # 状态管理
│   │   │   └── visualizationStore.ts  # Zustand store
│   │   └── types/             # TypeScript类型
│   │       └── index.ts       # 类型定义
│   ├── public/                # 静态资源
│   ├── package.json           # Node.js依赖
│   ├── tsconfig.json          # TypeScript配置
│   ├── tailwind.config.js     # Tailwind配置
│   ├── next.config.ts         # Next.js配置
│   └── README.md              # 前端文档
│
├── README.md                  # 项目主文档（英文）
├── README_zh.md               # 项目主文档（中文）
├── QUICKSTART.md              # 快速开始指南
└── .gitignore                 # Git忽略文件
```

## 🔧 API接口说明

### 1. 初始化会话

**端点**: `POST /api/init`

**请求体**:
```json
{
  "text": "Hello world",
  "config": {
    "n_vocab": 50257,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "d_k": 64,
    "max_seq_len": 512,
    "attention_type": "standard",
    "sparse_config": null
  }
}
```

**响应**:
```json
{
  "session_id": "uuid-string",
  "tokens": [15496, 995],
  "token_texts": ["Hello", "world"],
  "total_steps": 168,
  "initial_state": {
    "embeddings": [[...], [...]],
    "positional_encodings": [[...], [...]]
  }
}
```

### 2. 获取计算步骤

**端点**: `POST /api/step`

**请求体**:
```json
{
  "session_id": "uuid-string",
  "step": 0
}
```

**响应**:
```json
{
  "step": 0,
  "step_type": "layer_norm_1",
  "layer_index": 0,
  "description": "对输入进行Layer Normalization（注意力前）",
  "input_data": [[...], [...]],
  "output_data": [[...], [...]],
  "metadata": {
    "mean": [...],
    "variance": [...]
  }
}
```

### 3. 删除会话

**端点**: `DELETE /api/session/{session_id}`

**响应**:
```json
{
  "message": "会话已删除",
  "session_id": "uuid-string"
}
```

### 4. 列出所有会话

**端点**: `GET /api/sessions`

**响应**:
```json
{
  "total_sessions": 5,
  "session_ids": ["uuid-1", "uuid-2", ...]
}
```

## 🧮 计算步骤详解

### 每层Transformer的14个步骤

| 步骤 | 类型 | 说明 |
|------|------|------|
| 1 | `layer_norm_1` | 注意力前的Layer Normalization |
| 2 | `q_projection` | Query矩阵投影：Q = X @ W_q |
| 3 | `k_projection` | Key矩阵投影：K = X @ W_k |
| 4 | `v_projection` | Value矩阵投影：V = X @ W_v |
| 5 | `attention_scores` | 注意力分数：scores = (Q @ K^T) / √d_k |
| 6 | `attention_weights` | Softmax归一化 |
| 7 | `attention_output` | 加权求和：output = weights @ V |
| 8 | `attention_projection` | 输出投影：output @ W_o |
| 9 | `residual_1` | 残差连接：x = x + attention_output |
| 10 | `layer_norm_2` | FFN前的Layer Normalization |
| 11 | `ffn_hidden` | FFN第一层：hidden = x @ W1 + b1 |
| 12 | `ffn_relu` | ReLU激活函数 |
| 13 | `ffn_output` | FFN第二层：output = hidden @ W2 + b2 |
| 14 | `residual_2` | 残差连接：x = x + ffn_output |

### 输出层处理

| 步骤 | 说明 |
|------|------|
| `final_layer_norm` | 最终Layer Normalization |
| `token_selection` | 选择token（Next Token Prediction用最后一个） |
| `logits_head` | 投影到词汇表空间 |
| `softmax` | 转换为概率分布 |
| `prediction` | 选择最高概率的token |

## 📐 技术细节

### Transformer架构

本项目实现了标准的Transformer Decoder架构，包括：

**输入处理**:
- Token Embedding: 将token ID映射到高维向量空间
- Positional Encoding: 添加位置信息（正弦余弦编码）
- 组合: embedding + positional_encoding

**Transformer Block** (重复N层):
```
Input
  ↓
Layer Norm
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Layer Norm
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output (到下一层)
```

**输出层**:
```
Final Hidden State
  ↓
Layer Norm
  ↓
Select Token (last for LM)
  ↓
Linear (to vocab size)
  ↓
Softmax
  ↓
Predicted Token
```

### 多头注意力机制

**公式**:
```
Q = X @ W_q
K = X @ W_k  
V = X @ W_v

scores = (Q @ K^T) / √d_k
weights = softmax(scores)
output = weights @ V

MultiHead(X) = Concat(head_1, ..., head_h) @ W_o
```

**关键参数**:
- `n_head`: 注意力头数（如12）
- `d_k`: 每个头的维度（如64）
- `n_embd = n_head * d_k`: 模型维度（如768）

### 稀疏注意力

为了降低注意力机制的O(n²)复杂度，实现了多种稀疏模式：

**1. Sliding Window**
```
每个token只关注前后window_size个token
复杂度: O(n * window_size)
```

**2. Global-Local**
```
特定token（如[CLS]）可以被所有token关注
其他token使用局部注意力
```

**3. Blocked Attention**
```
将序列分成多个block
每个block内部全连接
```

**4. Random Attention**
```
随机选择一定比例的连接
保持稀疏性的同时引入随机性
```

### Mixture of Experts (MoE)

**架构**:
```
Input
  ↓
Router (Gating Network)
  ↓
Top-K Expert Selection
  ↓
Weighted Combination
  ↓
Output
```

**路由公式**:
```
G(x) = Softmax(x @ W_g)  # 门控网络
top_k_indices = TopK(G(x), k)
output = Σ G(x)[i] * Expert_i(x) for i in top_k_indices
```

**优势**:
- 模型容量大幅提升
- 计算成本相对较小
- 专家专门化学习

### 输出层处理

**1. Logits Head**
- 线性变换: `logits = hidden_state @ W_lm`
- 维度: `[batch, seq_len, n_vocab]`
- 无激活函数

**2. Softmax Temperature**
```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)

T > 1: 分布更平滑（更多样性）
T < 1: 分布更尖锐（更确定性）
T = 1: 标准Softmax
```

**3. 解码策略**
- **Greedy**: 总是选择概率最高的token
- **Top-K Sampling**: 从概率最高的K个中采样
- **Top-P (Nucleus) Sampling**: 从累积概率达到P的最小集合中采样
- **Beam Search**: 维护K个最可能的序列

## 🎓 教育资源

### 推荐论文

1. **Transformer原论文**
   - Vaswani et al. (2017) - "Attention is All You Need"
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **稀疏注意力**
   - Child et al. (2019) - "Generating Long Sequences with Sparse Transformers"
   - Beltagy et al. (2020) - "Longformer: The Long-Document Transformer"
   - Zaheer et al. (2020) - "Big Bird: Transformers for Longer Sequences"

3. **Mixture of Experts**
   - Shazeer et al. (2017) - "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
   - Fedus et al. (2021) - "Switch Transformers: Scaling to Trillion Parameter Models"
   - Lepikhin et al. (2021) - "GShard: Scaling Giant Models with Conditional Computation"

4. **大语言模型**
   - Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)
   - Touvron et al. (2023) - "LLaMA: Open and Efficient Foundation Language Models"

### 在线资源

- 📺 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- 📺 [Transformer可视化工具](https://transformer-circuits.pub/)
- 📚 [Hugging Face Course](https://huggingface.co/course)
- 💻 [Stanford CS224N](http://web.stanford.edu/class/cs224n/)

## 🐛 常见问题（FAQ）

### Q1: 为什么我的后端无法启动？

**A**: 请检查：
1. Python版本是否≥3.9
2. 是否已激活虚拟环境
3. 所有依赖是否已安装：`pip install -r requirements.txt`
4. 端口8000是否被占用

### Q2: 前端无法连接到后端？

**A**: 确保：
1. 后端服务正在运行（http://localhost:8000）
2. 检查CORS配置
3. 前端的API_BASE_URL配置正确

### Q3: 可视化动画不流畅？

**A**: 可能原因：
1. 序列长度过长，降低到<20
2. 浏览器性能不足，使用Chrome/Edge
3. 同时运行的步骤太多，减少层数

### Q4: 如何添加新的稀疏模式？

**A**: 
1. 在`backend/app/services/attention.py`中实现掩码生成函数
2. 在`SparseConfig`模型中添加新的模式类型
3. 更新前端UI以支持新模式的配置

### Q5: 支持加载预训练模型吗？

**A**: 当前版本使用随机初始化的权重，主要用于教育目的。如需加载预训练模型，可以：
1. 使用`transformers`库加载模型
2. 提取权重并传入可视化系统
3. 需要适配权重格式

### Q6: 如何部署到生产环境？

**A**: 参考部署指南：
1. 使用Gunicorn/Uvicorn作为WSGI服务器
2. Nginx作为反向代理
3. 使用Docker容器化部署
4. 配置SSL证书（Let's Encrypt）
5. 设置环境变量和密钥管理

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

1. **报告Bug**: 在GitHub Issues中提交详细的bug报告
2. **建议功能**: 提出新功能的想法和用例
3. **改进文档**: 修正错误、补充说明、翻译
4. **提交代码**: Fork项目，创建分支，提交Pull Request

### 开发流程

1. Fork本项目
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

### 代码规范

- **Python**: 遵循PEP 8，使用Black格式化
- **TypeScript**: 遵循Airbnb风格指南
- **提交信息**: 使用约定式提交（Conventional Commits）

## 📄 许可证

本项目采用 **MIT License** 开源。

详见 [LICENSE](LICENSE) 文件。

## 👥 作者与致谢

**项目维护者**: [您的名字]

**特别感谢**:
- Transformer原作者团队
- D3.js社区
- Next.js和FastAPI开发团队
- 所有贡献者和使用者

## 📞 联系方式

- **项目主页**: [GitHub Repository]
- **问题反馈**: [GitHub Issues]
- **邮件**: your-email@example.com
- **讨论社区**: [Discord/Slack链接]

## 🗺️ 项目路线图

### v1.0（当前版本）
- ✅ 基础Transformer可视化
- ✅ 多头注意力机制
- ✅ 稀疏注意力支持
- ✅ MoE可视化
- ✅ 输出层完整流程
- ✅ 中文文档

### v1.1（计划中）
- ⏳ 预训练模型加载
- ⏳ 更多解码策略（Beam Search, Top-P）
- ⏳ 批处理可视化
- ⏳ 性能优化

### v2.0（未来）
- 🔮 多语言支持（英文、日文等）
- 🔮 移动端适配
- 🔮 协作式学习功能
- 🔮 模型对比工具
- 🔮 自定义模型架构

## 🎉 开始探索吧！

现在你已经了解了项目的全部信息，开始你的Transformer可视化之旅吧！

```bash
# 一键启动（需先安装依赖）
cd backend && uvicorn app.main:app --reload &
cd frontend && npm run dev
```

访问 **http://localhost:3000**，享受学习的乐趣！

---

**⭐ 如果这个项目对你有帮助，请给我们一个Star！**

Happy Learning! 🚀
