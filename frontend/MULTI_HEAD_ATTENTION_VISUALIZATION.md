# Multi-Head Self-Attention D3.js Visualization

## 概述

本文档详细说明了使用 D3.js 和 TypeScript 实现的标准多头自注意力（Multi-Head Self-Attention）机制的完整可视化。

## 实现的功能

### ✅ 核心可视化步骤

#### 1. Layer Normalization (Pre-Norm)
- **输入矩阵显示**：彩色热力图展示输入数据 (n_tokens × n_embd)
- **归一化过程**：
  - 显示每个 token 的均值和方差计算
  - 标准化动画（颜色变化反映归一化过程）
  - 应用 gamma 和 beta 参数
- **输出显示**：归一化后的矩阵
- **交互功能**：悬停显示归一化前后的数值

#### 2. Q, K, V 矩阵生成
- **权重矩阵可视化**：
  - W_q, W_k, W_v 三个权重矩阵热力图
  - 不同颜色方案区分不同矩阵
  - 显示矩阵维度信息
- **矩阵乘法动画**：
  - 输入与权重矩阵相乘生成 Q, K, V
  - 渐进式动画展示计算过程
  - 交互式悬停查看具体数值

#### 3. 多头分割
- **可视化分割**：
  - 使用虚线框标识每个注意力头的区域
  - 不同颜色区分不同的头
  - 标注每个头的维度 (d_k = n_embd / n_head)
- **动画效果**：分割线逐个出现

#### 4. 注意力计算（每个头）
对前 3 个注意力头进行详细可视化：

##### 4.1 注意力得分矩阵
- **计算公式**：Attention_Score = Q @ K^T / sqrt(d_k)
- **可视化**：
  - 显示 Q 矩阵 (n_tokens × d_k)
  - K^T 矩阵 (d_k × n_tokens)
  - 矩阵乘法动画
  - 缩放操作 (÷√d_k)
- **热力图**：颜色深浅表示注意力得分强度
- **交互**：悬停显示具体得分值

##### 4.2 Softmax 归一化
- **逐行 Softmax**：每行转换为概率分布
- **可视化**：
  - 高亮当前处理的行
  - 颜色更新反映概率值
- **交互**：显示 Softmax 前后的值对比

##### 4.3 与 V 矩阵相乘
- **计算**：Output = Attention_Weights @ V
- **可视化**：
  - 显示注意力权重矩阵
  - 显示 V 矩阵
  - 矩阵乘法动画
  - 生成该头的输出矩阵

#### 5. 多头并行展示
- **布局设计**：
  - 为节省空间，详细展示前 3 个头
  - 其余头用文本说明表示
- **颜色编码**：每个头使用不同颜色
- **串行/并行模式**：可选择动画播放方式

#### 6. 多头输出拼接
- **可视化**：
  - 所有头的输出横向排列
  - 使用不同颜色标识每个头
  - 拼接动画效果
- **输出矩阵**：(n_tokens × n_embd)

#### 7. 输出线性变换 (W_o)
- **权重矩阵**：W_o 的热力图（显示采样部分）
- **矩阵乘法**：拼接后的矩阵与 W_o 相乘
- **最终输出**：注意力层输出矩阵

#### 8. 残差连接
- **可视化元素**：
  - 原始输入矩阵
  - 加号符号动画
  - 注意力输出矩阵
  - 元素级加法
- **最终输出**：残差连接后的结果
- **动画**：展示逐元素相加过程

## 组件结构

### MultiHeadAttentionViz

主可视化组件，包含完整的多头注意力机制可视化。

```typescript
interface MultiHeadAttentionVizProps {
  inputData: number[][];      // [n_tokens, n_embd]
  weights: {
    wq: number[][];           // [n_embd, n_embd]
    wk: number[][];           // [n_embd, n_embd]
    wv: number[][];           // [n_embd, n_embd]
    wo: number[][];           // [n_embd, n_embd]
    ln_gamma: number[];       // [n_embd]
    ln_beta: number[];        // [n_embd]
  };
  config: {
    n_head: number;
    d_k: number;
  };
  tokenTexts?: string[];
  animationMode?: 'serial' | 'parallel';
  onComplete?: () => void;
}
```

### MultiHeadAttentionDemo

演示组件，提供完整的用户界面和控制面板。

**功能特性**：
- 自动生成测试数据
- 播放控制（开始/重播）
- 动画模式切换（串行/并行）
- 配置信息显示
- 进度跟踪
- 完整的说明文档

## D3.js 技术实现

### 颜色方案

```typescript
// 不同头使用不同颜色
const headColors = d3.scaleOrdinal(d3.schemeCategory10);

// 权重矩阵颜色
const wqColor = d3.scaleSequential(d3.interpolateBlues);
const wkColor = d3.scaleSequential(d3.interpolateGreens);
const wvColor = d3.scaleSequential(d3.interpolatePurples);
const woColor = d3.scaleSequential(d3.interpolateViridis);

// 注意力分数
const scoreColor = d3.scaleSequential(d3.interpolateYlOrRd);

// 注意力权重（概率）
const weightColor = d3.scaleSequential(d3.interpolateBuPu);

// 数据矩阵
const dataColor = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);
```

### 热力图渲染

```typescript
const renderMatrix = (matrix, x, y, label, colorScaleFunc) => {
  const cells = g.selectAll('.cell')
    .data(matrix.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
    .join('rect')
    .attr('x', d => d.j * cellSize)
    .attr('y', d => d.i * cellSize)
    .attr('width', cellSize - 0.5)
    .attr('height', cellSize - 0.5)
    .attr('fill', d => colorScaleFunc(d.val));
  
  return { cells, width: cols * cellSize, height: rows * cellSize };
};
```

### 动画序列

```typescript
// 使用 async/await 控制动画流程
const runAnimation = async () => {
  // Step 1: Layer Normalization
  await animateLayerNorm();
  
  // Step 2: Generate Q, K, V
  await animateQKVGeneration();
  
  // Step 3: Split into heads
  await animateHeadSplit();
  
  // Step 4-N: Attention for each head
  for (let h = 0; h < nHead; h++) {
    await animateAttentionHead(h);
  }
  
  // Step N+1: Concatenate
  await animateConcatenation();
  
  // Step N+2: Output projection
  await animateOutputProjection();
  
  // Step N+3: Residual connection
  await animateResidualConnection();
};
```

### 交互式工具提示

```typescript
cells.on('mouseover', function(event, d) {
  d3.select(this)
    .attr('stroke', '#000')
    .attr('stroke-width', 2);

  const tooltip = g.append('g')
    .attr('class', 'tooltip')
    .attr('transform', `translate(${event.layerX}, ${event.layerY - 30})`);

  tooltip.append('rect')
    .attr('x', -50)
    .attr('y', -25)
    .attr('width', 100)
    .attr('height', 30)
    .attr('rx', 4)
    .attr('fill', '#2d3748')
    .style('opacity', 0.95);

  tooltip.append('text')
    .attr('text-anchor', 'middle')
    .attr('y', -5)
    .attr('font-size', 10)
    .attr('fill', 'white')
    .text(`[${d.i},${d.j}]: ${d.val.toFixed(3)}`);
})
.on('mouseout', function() {
  d3.select(this)
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5);
  g.selectAll('.tooltip').remove();
});
```

## 数学公式实现

### Layer Normalization

```typescript
const layerNorm = (input: number[][], gamma: number[], beta: number[]): number[][] => {
  return input.map(row => {
    const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
    const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
    const std = Math.sqrt(variance + 1e-5);
    return row.map((val, i) => ((val - mean) / std) * gamma[i] + beta[i]);
  });
};
```

### 矩阵乘法

```typescript
const matrixMultiply = (a: number[][], b: number[][]): number[][] => {
  const m = a.length;
  const n = b[0].length;
  const p = b.length;
  const result: number[][] = Array(m).fill(0).map(() => Array(n).fill(0));
  
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < p; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
};
```

### Softmax

```typescript
const softmax = (row: number[]): number[] => {
  const maxVal = Math.max(...row);
  const exps = row.map(x => Math.exp(x - maxVal));
  const sumExps = exps.reduce((sum, val) => sum + val, 0);
  return exps.map(exp => exp / sumExps);
};
```

## 使用示例

### 基础使用

```tsx
import { MultiHeadAttentionViz } from '@/components/visualizations';

const App = () => {
  const inputData = [...]; // [n_tokens, n_embd]
  const weights = {
    wq: [...],
    wk: [...],
    wv: [...],
    wo: [...],
    ln_gamma: [...],
    ln_beta: [...],
  };
  const config = {
    n_head: 8,
    d_k: 64,
  };

  return (
    <MultiHeadAttentionViz
      inputData={inputData}
      weights={weights}
      config={config}
      tokenTexts={['The', 'cat', 'sat']}
      animationMode="serial"
      onComplete={() => console.log('Complete!')}
    />
  );
};
```

### 使用演示组件

```tsx
import { MultiHeadAttentionDemo } from '@/components/visualizations';

const DemoPage = () => {
  return <MultiHeadAttentionDemo />;
};
```

## 性能优化

### 矩阵采样显示

对于大型权重矩阵（如 768×768），只显示采样部分（50×50）以提高渲染性能：

```typescript
const wqMatrix = renderMatrix(
  weights.wq.slice(0, 50).map(row => row.slice(0, 50)),
  x, y, 'W_q (sample)'
);
```

### 头数量限制

为了避免页面过长，只详细展示前 3 个注意力头：

```typescript
for (let h = 0; h < Math.min(nHead, 3); h++) {
  await animateAttentionHead(h);
}

if (nHead > 3) {
  g.append('text').text(`... (${nHead - 3} more heads omitted)`);
}
```

### 动画优化

使用 D3 的延迟和过渡来优化动画性能：

```typescript
cells.transition()
  .duration(800)
  .delay((d, i) => i * 2)  // 交错延迟
  .style('opacity', 1);
```

## 响应式设计

- **自适应宽度**：SVG 宽度根据容器动态调整
- **可滚动容器**：使用 `overflow-auto` 处理大型可视化
- **进度指示器**：顶部显示当前步骤和进度百分比

## 访问演示

### 开发模式

```bash
cd frontend
npm run dev
# 访问 http://localhost:3000/attention-demo
```

### 生产构建

```bash
npm run build
npm start
```

## 文件结构

```
frontend/
├── src/
│   ├── app/
│   │   └── attention-demo/
│   │       └── page.tsx                    # 演示页面路由
│   └── components/
│       └── visualizations/
│           ├── MultiHeadAttentionViz.tsx   # 主可视化组件
│           ├── MultiHeadAttentionDemo.tsx  # 演示组件
│           ├── index.ts                    # 导出
│           ├── examples.tsx                # 示例（包含注意力示例）
│           └── README.md                   # 组件文档
└── MULTI_HEAD_ATTENTION_VISUALIZATION.md   # 本文档
```

## 验收标准

- [x] Layer Normalization 正确可视化
- [x] Q, K, V 矩阵生成动画流畅
- [x] 矩阵乘法过程清晰展示
- [x] 注意力得分矩阵用热力图表示
- [x] Softmax 归一化过程可视化
- [x] 多个注意力头并行展示（前3个详细，其余省略）
- [x] 多头合并动画正确
- [x] 残差连接清晰可见
- [x] 所有矩阵元素可交互（悬停显示数值）
- [x] 性能优化（大矩阵采样显示）
- [x] 支持播放/暂停（通过重播按钮）
- [x] 进度跟踪和步骤指示
- [x] TypeScript 类型安全
- [x] 构建成功，无错误
- [x] Lint 通过（仅有安全的 ref 警告）

## 技术栈

- **D3.js v7**: SVG 操作和动画
- **React 19**: 组件框架
- **TypeScript**: 类型安全
- **Next.js 16**: 应用框架
- **Tailwind CSS**: 样式

## 已知限制

1. **头数量显示**：为节省空间和提高性能，只详细展示前 3 个注意力头
2. **矩阵采样**：大型权重矩阵（768×768）只显示 50×50 的采样
3. **简化计算**：某些中间步骤使用简化计算以提高演示流畅度
4. **屏幕空间**：完整可视化高度约 3000px，需要滚动查看

## 未来改进

- [ ] 添加暂停/继续按钮（当前只有重播）
- [ ] 添加单步执行模式
- [ ] 支持速度调节滑块
- [ ] 添加更详细的矩阵乘法逐元素动画
- [ ] 支持所有注意力头的展开/折叠查看
- [ ] 添加导出动画为 GIF/视频功能
- [ ] 添加交互式编辑权重矩阵
- [ ] 3D 可视化注意力模式

## 致谢

本实现基于标准的 Transformer 架构中的多头自注意力机制，参考了"Attention is All You Need"论文。

## 许可

本项目代码遵循项目整体许可证。
