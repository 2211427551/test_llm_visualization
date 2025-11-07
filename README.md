# Transformer 深度学习平台

## 📖 项目背景

本项目是一个基于 Transformer 架构的深度学习平台，专注于实现和展示先进的深度学习技术。项目集成了稀疏注意力机制（Sparse Attention）和混合专家模型（Mixture of Experts, MoE），提供了完整的从模型训练到推理部署的全栈解决方案。

### 🎯 核心价值

- **技术创新**：实现了业界领先的稀疏注意力和 MoE 技术
- **工程实践**：提供了完整的工程化实现和部署方案
- **教学演示**：通过可视化界面直观展示模型内部机制
- **研究平台**：为深度学习研究提供了灵活的实验环境

## ✨ 功能特性

### 🔬 核心技术

- **稀疏注意力机制**
  - 分组头注意力：局部注意力 + 全局注意力
  - 动态窗口大小：根据序列长度自适应调整
  - 数值稳定性：优化的掩码策略和梯度计算
  - 兼容性：与标准 Transformer 完全兼容

- **混合专家模型（MoE）**
  - Top-k 路由机制：智能选择最优专家组合
  - 负载均衡：确保专家利用率均衡
  - 可配置架构：支持不同规模的专家网络
  - 中间数据捕获：完整的路由过程可视化

- **高性能推理引擎**
  - FastAPI 后端：高性能异步 API 服务
  - PyTorch 优化：GPU 加速和内存优化
  - 批处理支持：高效的批量推理
  - 错误处理：完善的异常处理机制

### 🎨 可视化界面

- **三栏布局设计**：模型配置、实时推理、数据可视化
- **交互式操作**：直观的参数调整和结果展示
- **实时监控**：模型运行状态的实时反馈
- **响应式设计**：适配不同屏幕尺寸

## 🛠️ 技术栈

### 后端技术

| 技术 | 版本 | 说明 |
|------|------|------|
| Python | ^3.9 | 主要开发语言 |
| FastAPI | ^0.104.1 | 高性能 Web 框架 |
| PyTorch | ^2.1.0 | 深度学习框架 |
| Uvicorn | ^0.24.0 | ASGI 服务器 |
| Pydantic | ^2.5.0 | 数据验证和序列化 |
| Poetry | - | 依赖管理工具 |

### 前端技术

| 技术 | 版本 | 说明 |
|------|------|------|
| React | ^19.2.0 | 用户界面框架 |
| TypeScript | ~5.9.3 | 类型安全的 JavaScript |
| Vite | ^7.1.12 | 现代化构建工具 |
| Tailwind CSS | ^3.4.15 | 实用优先的 CSS 框架 |
| D3.js | ^7.9.0 | 数据可视化库 |
| i18next | ^25.6.0 | 国际化支持 |

### 部署技术

| 技术 | 说明 |
|------|------|
| Docker | 容器化部署 |
| Docker Compose | 多容器编排 |
| Nginx | 反向代理和静态文件服务 |
| Poetry | Python 依赖管理 |

## 📁 项目结构

```
.
├── backend/                 # 后端服务
│   ├── app/                # 应用核心代码
│   │   ├── core/           # 核心配置和中间件
│   │   ├── models/         # 深度学习模型
│   │   │   └── transformer/ # Transformer 模型实现
│   │   │       ├── sparse_attention.py  # 稀疏注意力
│   │   │       ├── moe.py               # MoE 实现
│   │   │       ├── attention.py         # 标准注意力
│   │   │       ├── block.py            # Transformer 块
│   │   │       └── model.py            # 完整模型
│   │   ├── routers/        # API 路由
│   │   ├── services/       # 业务逻辑服务
│   │   └── schemas/        # 数据模式定义
│   ├── tests/              # 测试文件
│   ├── pyproject.toml      # Python 项目配置
│   └── Dockerfile          # 后端容器配置
├── frontend/               # 前端应用
│   ├── src/               # 源代码
│   │   ├── components/    # React 组件
│   │   ├── hooks/         # 自定义 Hooks
│   │   ├── services/      # API 服务
│   │   ├── types/         # TypeScript 类型定义
│   │   └── utils/         # 工具函数
│   ├── public/            # 静态资源
│   ├── package.json       # 前端项目配置
│   └── Dockerfile         # 前端容器配置
├── deploy/                # 部署配置
│   ├── docker-compose.yml # 容器编排配置
│   ├── nginx/            # Nginx 配置
│   └── scripts/          # 部署脚本
├── docs/                 # 项目文档
├── .gitignore           # Git 忽略文件
├── .editorconfig        # 编辑器配置
└── README.md           # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.9 或更高版本
- **Node.js**: 18.0 或更高版本
- **Docker**: 20.10 或更高版本
- **Docker Compose**: 2.0 或更高版本

### WSL2 + Docker 环境配置

#### 1. 安装 WSL2

在 Windows 上安装 WSL2：

```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 下载并安装 WSL2 内核更新包
# 访问: https://aka.ms/wsl2kernel

# 设置 WSL2 为默认版本
wsl --set-default-version 2

# 安装 Ubuntu 发行版
wsl --install -d Ubuntu
```

#### 2. 安装 Docker Desktop

1. 下载 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 安装时确保启用 "Use WSL 2 based engine"
3. 在 WSL2 集成设置中启用所需的发行版

#### 3. 验证安装

```bash
# 在 WSL2 终端中验证
docker --version
docker-compose --version
docker run hello-world
```

### 本地开发环境启动

#### 方式一：Docker Compose（推荐）

```bash
# 克隆项目
git clone https://github.com/your-org/transformer-platform.git
cd transformer-platform

# 复制环境配置文件
cp deploy/.env.example deploy/.env

# 启动所有服务
cd deploy
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 方式二：本地开发

**后端启动：**

```bash
# 进入后端目录
cd backend

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell

# 启动开发服务器
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**前端启动：**

```bash
# 新开终端，进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 访问地址

- **前端应用**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/v1/health

## 📡 API 接口示例

### 1. 初始化模型

```bash
curl -X GET "http://localhost:8000/api/v1/initialize" \
  -H "Content-Type: application/json"
```

**响应示例：**
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
    "use_sparse_attention": false,
    "use_moe": false,
    "device": "cpu"
  }
}
```

### 2. 自定义配置初始化

```bash
curl -X GET "http://localhost:8000/api/v1/initialize?config={\"n_layer\":6,\"use_sparse_attention\":true,\"use_moe\":true}"
```

### 3. 文本推理

```bash
curl -X POST "http://localhost:8000/api/v1/forward" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "深度学习是人工智能的重要分支",
    "capture_data": true
  }'
```

**响应示例：**
```json
{
  "success": true,
  "message": "前向传播完成",
  "logits_shape": [1, 12, 50304],
  "sequence_length": 12,
  "captured_data": {
    "attention_weights": [...],
    "moe_routing": [...]
  }
}
```

### 4. 获取模型状态

```bash
curl -X GET "http://localhost:8000/api/v1/status"
```

## 🎨 前端预览说明

### 主界面布局

前端应用采用三栏布局设计：

1. **左面板** - 模型配置
   - 模型参数设置（层数、注意力头数、嵌入维度等）
   - 高级功能开关（稀疏注意力、MoE）
   - 实时配置验证

2. **中央面板** - 推理界面
   - 文本输入区域
   - 实时推理结果展示
   - 性能指标监控

3. **右面板** - 数据可视化
   - 注意力权重热力图
   - MoE 路由可视化
   - 模型内部状态分析

### 交互功能

- **参数调整**: 实时调整模型参数并查看效果
- **批量处理**: 支持多文本批量推理
- **结果导出**: 支持推理结果和可视化数据导出
- **主题切换**: 支持明暗主题切换

## 🧪 测试验证

### 后端测试

```bash
cd backend

# 运行所有测试
poetry run pytest

# 运行特定测试
poetry run pytest test_sparse_attention.py
poetry run pytest test_moe_unit.py

# 生成覆盖率报告
poetry run pytest --cov=app tests/
```

### 前端测试

```bash
cd frontend

# 运行单元测试
npm run test

# 运行测试并生成覆盖率报告
npm run test:coverage

# 运行 E2E 测试
npm run test:e2e
```

### 集成测试

```bash
cd deploy

# 运行部署测试
./test-deployment.sh

# 验证 API 集成
./validate-api.sh
```

## 📚 详细文档

- [稀疏注意力设计文档](docs/sparse-attention.md)
- [MoE 技术说明](docs/moe-design.md)
- [API 接口文档](docs/api-reference.md)
- [部署指南](docs/deployment-guide.md)
- [贡献指南](CONTRIBUTING.md)

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解详细信息。

### 开发流程

1. Fork 项目到个人仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 代码规范

- **Python**: 遵循 PEP 8，使用 Black 和 isort 格式化
- **TypeScript**: 遵循 ESLint 和 Prettier 配置
- **提交信息**: 使用约定式提交格式

## 📋 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新详情。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [NanoGPT](https://github.com/karpathy/nanoGPT) - Transformer 实现参考
- [Deepseek](https://github.com/deepseek-ai) - 稀疏注意力设计灵感
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [React](https://reactjs.org/) - 用户界面框架

## 📞 联系方式

- 项目主页: https://github.com/your-org/transformer-platform
- 问题反馈: https://github.com/your-org/transformer-platform/issues
- 邮箱: your-email@example.com

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！