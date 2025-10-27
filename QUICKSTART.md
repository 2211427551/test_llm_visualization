# 快速开始指南

## 项目概述

这是一个完整的 Transformer 计算可视化应用，包含：
- **后端**：Python + FastAPI (端口 8000)
- **前端**：Next.js 14 + TypeScript (端口 3000)

## 一键启动

### 终端 1：启动后端

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端运行在：http://localhost:8000
API文档：http://localhost:8000/docs

### 终端 2：启动前端

```bash
cd frontend
npm install
npm run dev
```

前端运行在：http://localhost:3000

## 使用流程

1. 打开浏览器访问 http://localhost:3000
2. 在输入框中输入文本，例如："Hello world"
3. （可选）点击"显示高级配置"调整模型参数
4. 点击"开始计算"按钮
5. 等待初始化完成（调用后端 API）
6. 使用控制面板浏览计算步骤：
   - 点击 ▶️ 自动播放
   - 使用 ⏮️ ⏭️ 单步前进/后退
   - 调整速度滑块改变播放速度
   - 使用下拉菜单跳转到特定步骤
7. 查看右侧解释面板了解当前步骤的详细信息
8. 可视化区域显示数据（JSON格式）

## 测试验证

运行测试脚本验证安装：

```bash
./test-frontend.sh
```

## 功能特性

### 输入模块
- 文本输入框
- 高级配置选项（词汇表大小、嵌入维度、层数等）
- 实时验证

### 控制面板
- 播放/暂停
- 单步前进/后退
- 速度控制 (0.5x - 3x)
- 步骤选择器
- 进度条

### 可视化画布
- Token 序列显示
- 当前步骤数据（JSON格式）
- 元数据展示
- D3.js 预留区域

### 解释面板
- 当前步骤说明
- 详细的中文解释（14种计算步骤）
- 数据形状信息
- 层信息

## 计算步骤说明

每个 Transformer 层包含 14 个计算步骤：

1. **layer_norm_1** - 注意力前的 Layer Normalization
2. **q_projection** - Query 矩阵投影
3. **k_projection** - Key 矩阵投影
4. **v_projection** - Value 矩阵投影
5. **attention_scores** - 注意力分数计算
6. **attention_weights** - Softmax 归一化
7. **attention_output** - 注意力输出
8. **attention_projection** - 输出投影
9. **residual_1** - 第一个残差连接
10. **layer_norm_2** - FFN 前的 Layer Normalization
11. **ffn_hidden** - 前馈网络第一层
12. **ffn_relu** - ReLU 激活
13. **ffn_output** - 前馈网络第二层
14. **residual_2** - 第二个残差连接

## 常见问题

### Q: 前端连接不上后端？
**A**: 检查：
1. 后端是否在 http://localhost:8000 运行
2. 检查 `frontend/.env.local` 中的 `NEXT_PUBLIC_API_URL` 配置
3. 查看浏览器控制台的网络请求

### Q: 页面样式不正确？
**A**: 
1. 清除浏览器缓存
2. 删除 `frontend/.next` 文件夹
3. 重新运行 `npm run dev`

### Q: npm install 失败？
**A**: 
1. 确保 Node.js 版本 >= 18
2. 尝试删除 `node_modules` 和 `package-lock.json`
3. 重新运行 `npm install`

### Q: TypeScript 编译错误？
**A**: 
1. 运行 `npm run build` 查看详细错误
2. 检查 `tsconfig.json` 配置
3. 确保所有依赖已正确安装

## 开发命令

### 后端
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload    # 开发服务器
python test_api.py               # 运行测试
```

### 前端
```bash
cd frontend
npm run dev      # 开发服务器
npm run build    # 生产构建
npm run start    # 生产服务器
npm run lint     # 代码检查
```

## 环境要求

### 后端
- Python 3.9+
- pip

### 前端
- Node.js 18+
- npm 或 yarn

## 下一步

- [ ] 集成 D3.js 进行交互式可视化
- [ ] 添加导出功能
- [ ] 实现会话保存/加载
- [ ] 优化性能
- [ ] 添加更多可视化类型

## 获取帮助

- 查看 `backend/README.md` 了解后端详情
- 查看 `frontend/README.md` 了解前端详情
- 查看 `frontend/SETUP.md` 了解详细设置步骤
- 查看 `frontend/IMPLEMENTATION_SUMMARY.md` 了解实现细节

## 许可证

MIT License
