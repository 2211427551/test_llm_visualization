# 前端工程

基于 **Vite + React + TypeScript** 的前端工程模板，内置中文本地化、Tailwind CSS 快速布局能力、D3.js 可视化示例以及统一的 ESLint/Prettier 代码风格。

## 技术栈

- 构建工具：Vite
- 前端框架：React 19 + TypeScript
- 样式方案：Tailwind CSS（支持深浅色主题）
- 本地化：i18next（默认中文文案，可扩展多语言）
- 可视化：D3.js（示例展示访问量趋势）
- 代码规范：ESLint + Prettier（集成 Tailwind CSS 排序插件）
- 测试：Vitest + Testing Library（提供基础 smoke test）

## 快速开始

```bash
# 安装依赖
npm install

# 启动开发环境
npm run dev

# 构建产物
npm run build

# 代码检查
npm run lint

# 代码格式化
npm run format

# 单元测试（一次性执行 / 监听模式）
npm run test
npm run test:watch
```

## 目录结构

```
frontend/
├── public/                  # 静态资源
├── src/
│   ├── components/          # 业务组件（含 D3 可视化示例）
│   ├── hooks/               # 自定义 Hooks（主题切换等）
│   ├── layouts/             # 页面布局组件
│   ├── services/            # 业务服务（i18n 配置等）
│   ├── styles/              # Tailwind CSS 入口与全局样式
│   ├── App.tsx              # 应用入口组件
│   └── main.tsx             # 渲染入口（注入 i18n、主题 Provider）
├── eslint.config.js         # ESLint 配置
├── prettier.config.js       # Prettier 配置
├── tailwind.config.js       # Tailwind 配置（深浅色主题、品牌色）
└── vite.config.ts           # Vite & Vitest 配置
```

## 主题与本地化

- 默认语言为简体中文，可在 `src/services/i18n.ts` 中扩展其它语言。
- `ThemeProvider` 会自动检测系统偏好并持久化用户选择，Tailwind 配置启用 `darkMode: 'class'` 以支持深浅色主题切换。

## 测试说明

仓库附带一个基础的烟雾测试（`src/App.test.tsx`），用于验证应用能够正确渲染核心文案及图表标题，可在 CI 中作为快速健康检查。
