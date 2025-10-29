# Sparse Attention D3.js Visualization

## 概述

本文档详细说明了使用 D3.js 和 TypeScript 实现的稀疏注意力（Sparse Attention）机制的完整可视化。这是本项目的核心创新点，用于展示现代LLM如何通过稀疏化提高效率。

## 实现的功能

### ✅ 核心可视化步骤

#### 1. 稀疏模式选择器

**UI组件**：
- 稀疏模式下拉菜单，支持以下选项：
  - "Dense (标准注意力)" - 无稀疏化
  - "Sliding Window (滑动窗口)" - 局部关注
  - "Global + Local (全局+局部)" - 混合关注模式
  - "Blocked (分块)" - 分块稀疏

**配置参数**：
- 窗口大小 (window_size): 1-5
- 块大小 (block_size): 2-6
- 全局token索引 (global_tokens): 自动设置为[0]

#### 2. 稀疏掩码可视化

##### 2.1 掩码矩阵生成
- 根据选择的稀疏模式自动生成掩码矩阵
- 矩阵大小: (n_tokens × n_tokens)
- 1表示允许关注，0表示屏蔽

##### 2.2 掩码模式展示
- **滑动窗口**: 对角线附近形成带状
- **全局+局部**: 第一行/列完全展开，其他行带状
- **分块**: 矩阵上出现块状结构
- **密集**: 完全填充

#### 3. 稀疏统计信息

显示关键指标：
- **稀疏度**: 被屏蔽连接的百分比
- **计算复杂度**: 从 O(n²) 降低到 O(n·w)
- **效率提升**: 实时计算并显示

#### 4. 稀疏注意力计算流程

##### 4.1 Layer Normalization
- 对输入进行归一化
- 与标准注意力相同

##### 4.2 Q, K, V 生成
- 通过权重矩阵 W_q, W_k, W_v 生成
- 使用热力图展示

##### 4.3 注意力分数计算（应用前）
- 计算: scores = Q @ K^T / √d_k
- 显示完整的注意力分数矩阵
- 使用渐变色表示得分强度

##### 4.4 应用稀疏掩码
- 被mask的位置设为 -∞ (实际为 -1e9)
- **掩码覆盖动画**:
  - 逐行扫描效果
  - 被mask的单元格变为黑色半透明
  - 延迟动画展示掩码应用过程

##### 4.5 Softmax（稀疏版本）
- 应用 Softmax 归一化
- 被mask的位置概率接近0
- 未被mask的位置重新归一化
- 使用紫蓝色渐变展示概率分布

##### 4.6 与 V 矩阵相乘
- 计算加权和: output = attention_weights @ V
- 由于稀疏性，很多 V 向量权重为0
- 展示最终的注意力输出

#### 5. 稀疏优势总结

可视化底部显示：
- ✓ 计算量减少 XX%
- ✓ 内存占用降低
- ✓ 支持更长序列

## 组件结构

### SparseAttentionViz

主可视化组件，实现稀疏注意力的完整流程。

```typescript
interface SparseAttentionVizProps {
  inputData: number[][];          // [n_tokens, n_embd]
  weights: {
    wq: number[][];               // [n_embd, n_embd]
    wk: number[][];               // [n_embd, n_embd]
    wv: number[][];               // [n_embd, n_embd]
    wo: number[][];               // [n_embd, n_embd]
    ln_gamma: number[];           // [n_embd]
    ln_beta: number[];            // [n_embd]
  };
  config: {
    n_head: number;
    d_k: number;
    sparse_pattern?: string;      // 稀疏模式
    window_size?: number;         // 窗口大小
    block_size?: number;          // 块大小
    global_tokens?: number[];     // 全局token索引
  };
  attentionMask?: number[][];     // 预计算的掩码矩阵
  tokenTexts?: string[];
  showComparison?: boolean;       // 是否显示对比
  onComplete?: () => void;
}
```

### SparseAttentionDemo

演示组件，提供完整的用户界面和控制面板。

**功能特性**：
- 稀疏模式选择器（下拉菜单）
- 参数调节（滑块）
  - 窗口大小调节
  - 块大小调节
- 对比模式切换（复选框）
- 播放/重播控制
- 实时统计显示
- 详细的模式说明

## D3.js 技术实现

### 颜色方案

```typescript
// 掩码矩阵（灰度）
const maskColor = d3.scaleSequential(d3.interpolateGreys).domain([0, 1]);

// 注意力分数（黄红渐变）
const scoreColor = d3.scaleSequential(d3.interpolateYlOrRd);

// 注意力权重（紫蓝渐变）
const weightColor = d3.scaleSequential(d3.interpolateBuPu).domain([0, 1]);

// 数据矩阵（红蓝发散）
const dataColor = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);
```

### 掩码覆盖层实现

```typescript
const maskCells = matrixGroup.selectAll('.mask-cell')
  .data(maskData.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
  .join('rect')
  .attr('class', 'mask-cell')
  .attr('x', d => d.j * cellSize)
  .attr('y', d => d.i * cellSize)
  .attr('width', cellSize - 0.5)
  .attr('height', cellSize - 0.5)
  .attr('fill', d => d.val === 0 ? '#000' : 'none')  // 黑色表示被mask
  .attr('stroke', 'none')
  .style('opacity', 0)
  .attr('pointer-events', 'none');  // 不干扰鼠标事件
```

### 掩码扫描动画

```typescript
// 逐行扫描效果
maskCells.transition()
  .duration(800)
  .delay((d, i) => Math.floor(i / nTokens) * 100)  // 按行延迟
  .style('opacity', 0.7);  // 半透明黑色
```

### 交互式工具提示（显示掩码状态）

```typescript
cells.on('mouseover', function(event, d) {
  const isMasked = maskData && maskData[d.i][d.j] === 0;
  
  tooltip.append('text')
    .text(isMasked 
      ? `[${d.i},${d.j}]: Masked` 
      : `[${d.i},${d.j}]: ${d.val.toFixed(3)}`
    );
});
```

## 稀疏模式实现

### 1. 滑动窗口（Sliding Window）

```typescript
for (let i = 0; i < nTokens; i++) {
  const start = Math.max(0, i - windowSize);
  const end = Math.min(nTokens, i + windowSize + 1);
  for (let j = start; j < end; j++) {
    mask[i][j] = 1;
  }
}
```

**视觉效果**: 对角线附近的带状区域
**复杂度**: O(n·w)，w为窗口大小

### 2. 全局+局部（Global + Local）

```typescript
// 先应用滑动窗口
generateSlidingWindowMask(nTokens, windowSize);

// 添加全局token关注
globalTokens.forEach(idx => {
  for (let j = 0; j < nTokens; j++) {
    mask[idx][j] = 1;  // 全局token关注所有
    mask[j][idx] = 1;  // 所有token关注全局token
  }
});
```

**视觉效果**: 某些行/列完全填充，其他行为带状
**复杂度**: O(n·w + g·n)，g为全局token数量

### 3. 分块（Blocked）

```typescript
const nBlocks = Math.ceil(nTokens / blockSize);
for (let i = 0; i < nTokens; i++) {
  const blockI = Math.floor(i / blockSize);
  
  // 关注当前块、前一块、后一块
  for (let blockJ = Math.max(0, blockI - 1); 
       blockJ <= Math.min(nBlocks - 1, blockI + 1); 
       blockJ++) {
    const start = blockJ * blockSize;
    const end = Math.min((blockJ + 1) * blockSize, nTokens);
    for (let j = start; j < end; j++) {
      mask[i][j] = 1;
    }
  }
}
```

**视觉效果**: 矩阵上的块状结构
**复杂度**: O(n·b)，b为块大小

## 使用示例

### 基础使用

```tsx
import { SparseAttentionViz } from '@/components/visualizations';

const App = () => {
  const inputData = [...];  // [n_tokens, n_embd]
  const weights = { wq, wk, wv, wo, ln_gamma, ln_beta };
  
  const config = {
    n_head: 4,
    d_k: 16,
    sparse_pattern: 'sliding_window',
    window_size: 3,
  };

  return (
    <SparseAttentionViz
      inputData={inputData}
      weights={weights}
      config={config}
      onComplete={() => console.log('完成')}
    />
  );
};
```

### 使用演示组件

```tsx
import { SparseAttentionDemo } from '@/components/visualizations';

const DemoPage = () => {
  return <SparseAttentionDemo />;
};
```

## 后端支持

### ModelConfig 扩展

```python
class SparseConfig(BaseModel):
    pattern: Literal["dense", "sliding_window", "global_local", "blocked", "random", "custom"]
    window_size: Optional[int] = 3
    block_size: Optional[int] = 4
    global_tokens: Optional[List[int]] = None
    random_ratio: Optional[float] = 0.1
    custom_mask: Optional[List[List[int]]] = None

class ModelConfig(BaseModel):
    # ... 其他字段
    attention_type: Literal["standard", "sparse"] = "standard"
    sparse_config: Optional[SparseConfig] = None
```

### 稀疏掩码生成工具

位于 `backend/app/utils/sparse_mask.py`:

- `generate_sliding_window_mask()`
- `generate_global_local_mask()`
- `generate_blocked_mask()`
- `generate_random_mask()`
- `generate_longformer_mask()`
- `calculate_sparsity()`
- `apply_mask_to_scores()`

## 性能优化

### 1. 矩阵采样
对于大型矩阵，只显示关键部分：
```typescript
const sample = matrix.slice(0, 50).map(row => row.slice(0, 50));
```

### 2. 动画延迟优化
使用交错延迟提升视觉效果：
```typescript
.delay((d, i) => Math.floor(i / nTokens) * 100)  // 按行延迟
```

### 3. 头数限制
只详细展示第一个头的计算过程，节省渲染时间。

## 稀疏度指标

### 计算公式

```typescript
const calculateSparsity = (mask: number[][]): number => {
  const total = mask.length * mask[0].length;
  const masked = mask.flat().filter(v => v === 0).length;
  return masked / total;  // 被屏蔽的比例
};
```

### 典型稀疏度值

| 模式 | 序列长度 | 窗口/块大小 | 稀疏度 |
|------|----------|-------------|--------|
| Dense | 8 | - | 0% |
| Sliding Window | 8 | 3 | ~50-60% |
| Global + Local | 8 | 3 | ~40-50% |
| Blocked | 8 | 4 | ~40-50% |

## 文件结构

```
frontend/
├── src/
│   ├── app/
│   │   └── sparse-attention-demo/
│   │       └── page.tsx                          # 演示页面路由
│   ├── components/
│   │   └── visualizations/
│   │       ├── SparseAttentionViz.tsx            # 主可视化组件
│   │       ├── SparseAttentionDemo.tsx           # 演示组件
│   │       └── index.ts                          # 导出
│   └── types/
│       └── index.ts                              # 类型定义（含SparseConfig）
└── SPARSE_ATTENTION_VISUALIZATION.md             # 本文档

backend/
└── app/
    ├── utils/
    │   ├── config.py                             # 配置（含SparseConfig）
    │   └── sparse_mask.py                        # 稀疏掩码生成
    └── models/
        └── response.py                           # API响应（含attention_mask）
```

## 验收标准

- [x] 支持多种稀疏模式（滑动窗口、全局+局部、分块、密集）
- [x] 稀疏掩码正确生成并可视化
- [x] 掩码矩阵以热力图形式展示
- [x] 掩码覆盖层动画流畅（逐行扫描）
- [x] 被mask的单元格视觉上明显区分（黑色半透明）
- [x] 显示稀疏度统计信息
- [x] 交互式悬停显示"Masked"状态
- [x] 支持参数动态调节（窗口大小、块大小）
- [x] 与控制面板集成（模式选择器）
- [x] 提供详细的模式说明文档
- [x] TypeScript 类型安全
- [x] 后端配置扩展完成
- [x] 稀疏掩码生成工具实现

## 技术栈

- **D3.js v7**: SVG 操作和动画
- **React 19**: 组件框架
- **TypeScript**: 类型安全
- **Next.js 16**: 应用框架
- **Tailwind CSS**: 样式
- **Python**: 后端（FastAPI）
- **NumPy**: 矩阵计算

## 创新点

1. **直观的掩码可视化**: 使用黑色半透明覆盖层清晰展示被屏蔽的连接
2. **逐行扫描动画**: 动态展示掩码应用过程，增强教学效果
3. **实时稀疏度计算**: 量化展示效率提升
4. **多模式支持**: 涵盖主流稀疏注意力模式
5. **交互式参数调节**: 实时调整窗口大小等参数，立即看到效果
6. **教学友好**: 详细的文档和说明，适合学习和演示

## 应用场景

### 教学
- 理解稀疏注意力的工作原理
- 对比密集与稀疏注意力的差异
- 学习不同稀疏模式的特点

### 研究
- 快速原型新的稀疏模式
- 可视化验证稀疏策略
- 分析不同模式的效率权衡

### 演示
- 向非技术人员解释稀疏注意力
- 展示 LLM 效率优化技术
- 会议和讲座的可视化工具

## 未来改进

- [ ] 添加更多稀疏模式（Random、Longformer风格等）
- [ ] 实现密集vs稀疏并排对比视图
- [ ] 添加自定义掩码编辑功能
- [ ] 支持多头稀疏注意力展示
- [ ] 添加注意力路径高亮（点击token查看连接）
- [ ] 3D可视化稀疏模式
- [ ] 性能基准测试对比
- [ ] 导出掩码矩阵为图片/JSON

## 参考文献

- Longformer: The Long-Document Transformer (Beltagy et al., 2020)
- Big Bird: Transformers for Longer Sequences (Zaheer et al., 2020)
- Generating Long Sequences with Sparse Transformers (Child et al., 2019)
- Reformer: The Efficient Transformer (Kitaev et al., 2020)

## 许可

本项目代码遵循项目整体许可证。
