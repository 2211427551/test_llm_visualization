# MoE Feed-Forward Network Visualization

## 概述

混合专家模型（Mixture of Experts, MoE）前馈网络可视化是本项目的核心创新点之一。该可视化完整展示了MoE架构如何通过门控网络将不同的token路由到不同的专家网络，实现条件计算和模型容量扩展。

## 技术栈

- **React 19** with TypeScript
- **D3.js v7** - 数据驱动的动画和交互
- **Next.js 16** - 现代React框架
- **Tailwind CSS 4** - 样式系统

## 组件架构

### 主要组件

1. **MoEFFNViz** (`MoEFFNViz.tsx`)
   - 核心可视化组件
   - 完整的MoE FFN流程动画
   - D3.js驱动的交互式可视化

2. **MoEFFNDemo** (`MoEFFNDemo.tsx`)
   - 演示和教学用组件
   - 包含控制面板和说明文档
   - 生成示例数据

## 可视化流程

### 1. Layer Normalization (Pre-Norm)

对输入进行归一化处理，为后续的门控计算做准备。

**可视化元素：**
- 输入矩阵热力图
- 归一化后的矩阵热力图
- 渐进式单元格动画

### 2. 门控网络 (Gating Network)

计算每个token选择各个专家的logits值。

**数学公式：**
```
logits = normalized_input @ gate_weights
```

**可视化元素：**
- 门控权重矩阵 (n_embd × n_experts)
- logits矩阵
- 使用Viridis配色方案

### 3. Softmax归一化

将logits转换为概率分布。

**数学公式：**
```
probs[i] = exp(logits[i] - max(logits)) / sum(exp(logits - max(logits)))
```

**可视化元素：**
- 专家概率矩阵
- 使用BuPu配色方案（0到1）

### 4. 专家选择可视化

展示每个token对各个专家的选择概率。

**可视化方式：**

#### 条形图 (Bar Chart)
- 为每个token独立显示一个条形图
- X轴：专家编号 (Expert 0, 1, 2, ...)
- Y轴：选择概率
- Top-K专家用专家特定颜色高亮
- 未选中的专家显示为灰色
- 悬停显示精确概率值

**交互功能：**
- 鼠标悬停：高亮显示单个条形
- 显示精确概率值

### 5. 专家网络布局

并排展示所有可用的专家网络。

**布局特点：**
- 网格布局（默认每行4个专家）
- 每个专家独立的矩形框
- 专家编号和颜色编码
- 显示网络结构：
  - Linear (n_embd → d_ff)
  - ReLU激活
  - Linear (d_ff → n_embd)
- 权重矩阵缩略图

**颜色方案：**
- 使用D3的Category10配色
- 每个专家有独特的颜色标识

### 6. 路由动画

展示token如何被路由到选中的专家。

**动画效果：**
- Token显示为圆形节点
- 从token到专家绘制贝塞尔曲线
- 线条粗细反映门控权重
- 线条颜色对应专家颜色
- 带箭头标记
- 串行动画（token逐个路由）
- 选中的专家高亮并产生发光效果

**路径生成：**
```typescript
const pathData = [
  { x: tokenX, y: tokenY },
  { x: tokenX, y: controlY1 },
  { x: expertX, y: controlY2 },
  { x: expertX, y: expertY }
];

const lineGenerator = d3.line()
  .x(d => d.x)
  .y(d => d.y)
  .curve(d3.curveBasis);
```

### 7. 专家内部计算

展示专家网络内部的FFN计算过程。

**计算流程：**

#### 第一层：Linear + ReLU
```
hidden = ReLU(input @ W1)
```

**可视化：**
- 隐藏层向量（应用ReLU后）
- 使用绿色配色表示激活值
- 渐进式动画

#### 第二层：Linear
```
output = hidden @ W2
```

**可视化：**
- 专家输出向量
- 使用RdBu配色（红蓝发散）

**示例展示：**
- 默认展示Token 0的前2个选中专家
- 清晰标注门控权重值

### 8. 输出加权与合并

将各个专家的输出按门控权重加权求和。

**数学公式：**
```
final_output[token] = Σ (gate_weight[expert_i] × expert_output[expert_i])
                      for expert_i in top_k_experts
```

**可视化：**
- MoE输出矩阵
- 展示所有token的合并输出

### 9. 残差连接

将原始输入加到MoE输出上。

**数学公式：**
```
final = input + moe_output
```

**可视化：**
- 最终输出矩阵
- 元素级加法动画

### 10. 专家负载均衡统计

实时统计每个专家被选中的次数。

**可视化元素：**
- 条形图展示使用频率
- X轴：专家编号
- Y轴：被选中次数
- 颜色编码：
  - 使用率低（<30%最大值）：红色警告
  - 正常使用：专家特定颜色
- 显示具体使用次数

**负载均衡分析：**
- 理想情况：所有专家均衡使用
- 问题指示：某些专家很少被使用

## API接口

### MoEFFNViz Props

```typescript
interface MoEFFNVizProps {
  inputData: number[][];  // [n_tokens, n_embd] - 输入token向量
  weights: {
    ln_gamma: number[];   // [n_embd] - Layer Norm缩放参数
    ln_beta: number[];    // [n_embd] - Layer Norm偏移参数
    gate_weights: number[][];  // [n_embd, n_experts] - 门控权重矩阵
    experts: {
      w1: number[][];  // [n_embd, d_ff] - 专家第一层权重
      w2: number[][];  // [d_ff, n_embd] - 专家第二层权重
    }[];  // 所有专家的权重
  };
  config: {
    n_experts: number;  // 专家总数
    top_k: number;      // 每个token选择的专家数量
    d_ff: number;       // FFN隐藏层维度（通常是4 × n_embd）
    n_embd: number;     // 嵌入维度
  };
  tokenTexts?: string[];  // token文本（可选，用于显示）
  animationMode?: 'serial' | 'parallel';  // 动画模式
  onComplete?: () => void;  // 完成回调
}
```

### MoEFFNDemo

演示组件，无需props，内部生成示例数据。

## 使用示例

### 基础使用

```tsx
import { MoEFFNViz } from '@/components/visualizations';

export default function MyPage() {
  const inputData = [
    [0.1, -0.2, 0.3, ...],  // Token 0
    [0.2, 0.1, -0.1, ...],  // Token 1
  ];

  const weights = {
    ln_gamma: [...],
    ln_beta: [...],
    gate_weights: [...],
    experts: [
      { w1: [...], w2: [...] },  // Expert 0
      { w1: [...], w2: [...] },  // Expert 1
      // ...
    ],
  };

  const config = {
    n_experts: 8,
    top_k: 2,
    d_ff: 64,
    n_embd: 16,
  };

  return (
    <MoEFFNViz
      inputData={inputData}
      weights={weights}
      config={config}
      tokenTexts={['The', 'cat']}
      onComplete={() => console.log('完成')}
    />
  );
}
```

### 使用Demo组件

```tsx
import { MoEFFNDemo } from '@/components/visualizations';

export default function DemoPage() {
  return <MoEFFNDemo />;
}
```

## 交互功能

### 鼠标悬停 (Hover)

- **矩阵单元格**：显示精确数值
- **条形图**：高亮并加粗边框
- **专家网络**：（可扩展）显示详细参数

### 动画控制

- **进度条**：实时显示可视化进度
- **步骤描述**：当前正在执行的步骤
- **自动播放**：组件挂载后自动开始动画

## 性能优化

### 1. 矩阵采样
- 对于大型权重矩阵，只显示子集（如50×50）
- 向量显示时限制最大长度（如100维）

### 2. 动画延迟优化
- 使用合理的延迟策略避免过长等待
- 批量元素的动画延迟适度分散

### 3. D3过渡管理
- 清理旧的过渡效果
- 在组件卸载时清理SVG元素

### 4. 专家数量适配
- 自动调整网格布局
- 支持4-32个专家的可视化

## 样式定制

### 颜色方案

```typescript
// 专家颜色
const expertColors = d3.scaleOrdinal(d3.schemeCategory10);

// 矩阵热力图
const colorScale = d3.scaleSequential(d3.interpolateRdBu)
  .domain([1, -1]);

// 概率图
const probScale = d3.scaleSequential(d3.interpolateBuPu)
  .domain([0, 1]);
```

### 尺寸参数

```typescript
const cellSize = 6;              // 矩阵单元格大小
const expertWidth = 180;         // 专家框宽度
const expertHeight = 200;        // 专家框高度
const barChartHeight = 120;      // 条形图高度
```

## 教育意义

### 核心概念展示

1. **条件计算**：不同token使用不同的计算路径
2. **稀疏激活**：每个token只激活top-k个专家
3. **容量扩展**：在不大幅增加计算的情况下增加模型参数
4. **专家专业化**：不同专家学习不同的模式

### 适用场景

- **研究人员**：理解MoE架构细节
- **学生**：学习Transformer变体
- **工程师**：调试和优化MoE模型
- **展示**：向非技术人员解释MoE

## 扩展方向

### 已实现功能 ✅
- [x] Layer Normalization可视化
- [x] 门控网络计算
- [x] Softmax概率分布
- [x] 条形图展示专家选择
- [x] Top-K专家高亮
- [x] 路由动画
- [x] 专家网络布局
- [x] 专家内部计算
- [x] 输出加权与合并
- [x] 残差连接
- [x] 负载均衡统计

### 可选扩展功能 🔮

- [ ] 热力图模式（tokens × experts）
- [ ] 专家折叠/展开功能
- [ ] 全景视图/聚焦视图切换
- [ ] 并行路由动画模式
- [ ] 专家权重3D可视化
- [ ] 负载均衡损失函数可视化
- [ ] 辅助损失（auxiliary loss）展示
- [ ] 专家容量限制可视化
- [ ] Token溢出处理展示
- [ ] 噪声门控（Noisy Gating）机制
- [ ] 批次级别的路由统计
- [ ] 导出可视化为视频/GIF

## 常见问题

### Q: 如何调整专家数量？
A: 修改`config.n_experts`参数，组件会自动调整布局。建议值：4-32。

### Q: 动画太快/太慢怎么办？
A: 修改各个transition的duration值，位于`runAnimation`函数中。

### Q: 如何支持更多token？
A: 组件理论上支持任意数量token，但建议演示时使用1-5个token以保证清晰度。

### Q: 可以只显示部分步骤吗？
A: 可以修改`runAnimation`函数，注释掉不需要的步骤。

### Q: 如何集成到现有项目？
A: 导入组件并传入符合接口的数据即可，参考"使用示例"部分。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [D3.js Documentation](https://d3js.org/)

## 贡献者

欢迎提交Issue和Pull Request来改进这个可视化！

## 许可证

遵循项目整体许可证。
