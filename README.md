# LLM 高级可视化

一个用于可视化和分析大语言模型的综合性 Web 应用程序，具有高级交互组件。

## 项目结构

```
llm-viz-advanced/
├── backend/                 # FastAPI Python 后端
│   ├── app/                # 应用程序包
│   │   ├── __init__.py
│   │   └── main.py         # FastAPI 应用程序入口点
│   ├── requirements.txt    # Python 依赖项
│   └── Dockerfile         # Docker 配置
├── frontend/               # Svelte + TypeScript 前端
│   ├── src/
│   │   ├── routes/        # SvelteKit 路由
│   │   ├── app.html       # HTML 模板
│   │   └── app.css        # 全局样式
│   ├── static/            # 静态资源
│   ├── package.json       # Node.js 依赖项
│   ├── svelte.config.js   # SvelteKit 配置
│   ├── tsconfig.json      # TypeScript 配置
│   ├── vite.config.ts     # Vite 配置
│   ├── tailwind.config.js # Tailwind CSS 配置
│   └── postcss.config.js  # PostCSS 配置
└── README.md              # 本文件
```

## 开发环境设置

### 后端设置

1. 导航到后端目录：
   ```bash
   cd backend
   ```

2. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

4. 运行开发服务器：
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

后端 API 将在 `http://localhost:8000` 上可用

### 前端设置

1. 导航到前端目录：
   ```bash
   cd frontend
   ```

2. 安装依赖项：
   ```bash
   npm install
   ```

3. 运行开发服务器：
   ```bash
   npm run dev
   ```

前端应用程序将在 `http://localhost:5173` 上可用

### Docker 设置

#### 后端 Docker

1. 导航到后端目录：
   ```bash
   cd backend
   ```

2. 构建并运行 Docker 容器：
   ```bash
   docker build -t llm-viz-backend .
   docker run -p 8000:8000 llm-viz-backend
   ```

## 技术栈

### 后端
- **FastAPI**: 用于构建 API 的现代、快速的 Web 框架
- **Uvicorn**: 用于运行 FastAPI 应用程序的 ASGI 服务器
- **PyTorch**: 机器学习框架（占位符）
- **Transformers**: Transformer 模型的 NLP 库（占位符）
- **NumPy & Pandas**: 数据处理库
- **Matplotlib & Plotly**: 可视化库

### 前端
- **SvelteKit**: 基于 Svelte 构建的全栈 Web 框架
- **TypeScript**: 类型安全的 JavaScript
- **Tailwind CSS**: 实用优先的 CSS 框架
- **Vite**: 构建工具和开发服务器
- **D3.js**: 数据可视化库

## API 端点

- `GET /` - 根端点
- `GET /health` - 健康检查端点

## 贡献

本项目目前正在开发中。请参考项目文档了解贡献指南。

## 许可证

[在此处添加许可证信息]