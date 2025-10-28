# Transformer 计算可视化 - 前端

这是一个基于 Next.js 14 的前端应用，用于可视化 Transformer 模型的计算过程。

## 技术栈

- **Next.js 14+** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **Zustand** (状态管理)
- **Axios** (HTTP 客户端)

## 项目结构

```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx           # 主页面
│   │   ├── layout.tsx         # 根布局
│   │   └── globals.css        # 全局样式
│   ├── components/
│   │   ├── InputModule.tsx    # 输入模块
│   │   ├── ControlPanel.tsx   # 控制面板
│   │   ├── VisualizationCanvas.tsx  # 可视化画布
│   │   └── ExplanationPanel.tsx     # 解释面板
│   ├── store/
│   │   └── visualizationStore.ts    # Zustand状态管理
│   ├── services/
│   │   └── api.ts             # API调用封装
│   └── types/
│       └── index.ts           # TypeScript类型定义
├── public/
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
```

## 开始使用

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

创建 `.env.local` 文件：

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. 启动开发服务器

```bash
npm run dev
```

应用将在 [http://localhost:3000](http://localhost:3000) 运行。

### 4. 构建生产版本

```bash
npm run build
npm start
```

## 功能特性

### 输入模块 (InputModule)
- 文本输入框
- 高级配置选项（词汇表大小、嵌入维度、层数等）
- 初始化计算按钮

### 控制面板 (ControlPanel)
- ▶️ 播放/暂停按钮
- ⏮️ 单步后退按钮
- ⏭️ 单步前进按钮
- 速度控制滑块 (0.5x - 3x)
- 步骤选择器
- 进度条

### 可视化画布 (VisualizationCanvas)
- Token 序列展示
- 当前步骤数据显示
- JSON 数据格式化展示
- D3.js 集成预留区域

### 解释面板 (ExplanationPanel)
- 当前步骤说明
- 详细计算解释
- 数据形状信息
- 层信息展示

## 状态管理

使用 Zustand 管理以下状态：
- 会话信息 (sessionId)
- 计算状态 (isInitialized, isPlaying, currentStep, etc.)
- 数据 (inputText, tokens, currentStepData)
- 配置 (config)
- 错误处理 (error)

## API 集成

后端 API 端点：
- `POST /api/init` - 初始化计算
- `POST /api/step` - 获取步骤数据
- `DELETE /api/session/{session_id}` - 删除会话

## 响应式设计

- **大屏**: 左右布局（可视化 70% / 解释 30%）
- **小屏**: 上下堆叠布局

## 开发脚本

```bash
npm run dev      # 启动开发服务器
npm run build    # 构建生产版本
npm run start    # 启动生产服务器
npm run lint     # 运行 ESLint
```

## 注意事项

- 确保后端服务在 `http://localhost:8000` 运行
- 此阶段不包含 D3.js 可视化实现
- 可视化区域目前显示 JSON 格式的数据

## 后续开发

- 集成 D3.js 进行交互式可视化
- 添加更多可视化类型（矩阵热图、注意力权重等）
- 优化性能和用户体验
