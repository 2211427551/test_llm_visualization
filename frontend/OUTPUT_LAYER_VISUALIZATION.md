# 输出层可视化文档

## 概述

输出层可视化组件展示了Transformer模型从最终隐藏状态到预测结果的完整流程，包括Layer Normalization、Token选择、Logits Head投影、Softmax归一化和最终预测。

## 组件架构

### 主组件：OutputLayerViz

```typescript
interface OutputLayerVizProps {
  finalHiddenState: number[][];  // 最后一层Transformer的输出 [seq_len, n_embd]
  vocabulary: string[];          // 词汇表
  onComplete?: () => void;       // 完成回调
}
```

### 子组件

1. **FinalLayerNormViz** - 最终Layer Normalization可视化
2. **TokenSelectionViz** - Token选择策略可视化
3. **LogitsHeadViz** - Logits Head投影层可视化
4. **SoftmaxViz** - Softmax归一化可视化
5. **PredictionDisplay** - 预测结果展示
6. **TopKPredictions** - Top-K候选列表

## 可视化流程

### 阶段1: Final Layer Normalization

**目的**: 对最后一个Transformer Block的输出进行归一化

**可视化元素**:
- 矩阵热图显示归一化后的向量
- 每个token的向量用一行表示
- 颜色编码：蓝-白-红（负值到正值）
- 渐进式动画（从透明到不透明）

**实现细节**:
```typescript
// 归一化值映射到颜色
const normalizedVal = Math.tanh(val);
const color = d3.interpolateRdBu(0.5 - normalizedVal * 0.5);
```

### 阶段2: Token Selection

**目的**: 根据任务类型选择合适的token

**支持的策略**:
- **Next Token Prediction**: 使用最后一个token（语言建模）
- **Classification**: 使用第一个token [CLS]（分类任务）
- **Sequence Labeling**: 使用所有token（序列标注）

**可视化元素**:
- 所有token以矩形框显示
- 选中的token（最后一个）高亮显示：
  - 绿色填充 (#10b981)
  - 绿色边框加粗
  - 光晕效果 (drop-shadow)
  - 标注文字 "← 用于预测"

**动画效果**:
```typescript
box.transition()
  .duration(800)
  .attr('opacity', 1)
  .attr('filter', 'drop-shadow(0 0 10px #10b981)');
```

### 阶段3: Logits Head

**目的**: 将选中token的向量投影到词汇表空间

**数学公式**:
```
logits = hidden_state @ W_lm
其中: W_lm 形状为 [n_embd, n_vocab]
```

**显示策略**:
由于词汇表通常很大（50,000+），只显示Top-K（默认20）个最高的logits

**可视化元素**:
- X轴：token文本（倾斜45度显示）
- Y轴：logit值（可正可负）
- 柱状图：
  - 正值用橙色 (#f97316)
  - 负值用蓝色 (#3b82f6)
  - 高度表示logit的绝对值
- 零线：清晰标注

**交互功能**:
```typescript
// 悬停显示详细信息
bars.on('mouseover', (event, d) => {
  // 显示tooltip:
  // - Token文本
  // - Logit值
  // - 高亮当前柱
});
```

**动画**:
```typescript
bars.transition()
  .duration(1000)
  .attr('y', d => Math.min(yScale(d.logit), yScale(0)))
  .attr('height', d => Math.abs(yScale(d.logit) - yScale(0)));
```

### 阶段4: Softmax归一化

**目的**: 将logits转换为概率分布

**数学公式**:
```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```

**可视化元素**:
- 从logits柱状图平滑过渡到概率柱状图
- Y轴从 "Logit值" 变为 "概率"
- Y轴刻度从实数值变为百分比
- 柱状图高度重新缩放到[0, 1]范围
- 颜色渐变（Viridis配色）：
  - 低概率：深紫色
  - 高概率：亮黄色

**动画效果**:
```typescript
// 1500ms的平滑过渡
bars.transition()
  .duration(1500)
  .attr('y', d => yScale(d.probability))
  .attr('height', d => yScale(0) - yScale(d.probability));

// Y轴标签更新
yAxisLabel.transition()
  .duration(1500)
  .text('概率');
```

**交互增强**:
- 悬停显示：
  - Token文本
  - 概率（百分比）
  - 原始Logit值（对比）

### 阶段5: 最终预测

**目的**: 展示预测结果和候选token

**主要元素**:

1. **预测结果卡片**（顶部）:
   ```
   ┌─────────────────────────────────┐
   │  ✓  预测结果: world             │
   │     概率: 45.23%                │
   └─────────────────────────────────┘
   ```
   - 金色背景渐变
   - 金色边框
   - 放大显示预测token
   - 显示置信度（概率）

2. **概率分布柱状图**:
   - 最高概率的token用金色 (#fbbf24) 高亮
   - 金色边框和光晕效果
   - 其他token用Viridis配色
   - 在每个柱上方显示概率百分比

3. **高亮效果**:
   ```typescript
   bars.filter((d, i) => i === 0)  // 第一个（最高概率）
     .transition()
     .duration(800)
     .attr('stroke', '#f59e0b')
     .attr('stroke-width', 3)
     .attr('opacity', 1)
     .style('filter', 'drop-shadow(0 0 15px #fbbf24)');
   ```

### 附加组件: Top-K预测列表

**目的**: 以列表形式展示Top-10候选token

**布局**:
```
1. world      [============================] 45.23%  ✓
2. there      [==========                  ] 12.85%
3. again      [======                      ]  8.47%
4. here       [====                        ]  5.32%
...
```

**特性**:
- 第一名（预测结果）：
  - 金色背景
  - 金色边框
  - ✓标记
  - 文字放大
- 其他候选：
  - 灰色背景
  - 蓝色进度条
- 进度条宽度对应概率
- 显示准确的百分比

## 搜索功能

### 实现

```typescript
const [searchTerm, setSearchTerm] = useState('');

const filteredPredictions = searchTerm
  ? predictions.filter(p => 
      p.token.toLowerCase().includes(searchTerm.toLowerCase())
    )
  : predictions;
```

### 界面

```html
<input
  type="text"
  placeholder="搜索token..."
  className="w-full px-4 py-2 border rounded-lg"
  value={searchTerm}
  onChange={(e) => setSearchTerm(e.target.value)}
/>
```

### 用途

- 查找特定词汇的概率
- 过滤显示的token
- 实时更新可视化

## 配色方案

### Logits阶段
- **正值**: 橙色系 (#f97316, #fb923c)
- **负值**: 蓝色系 (#3b82f6, #60a5fa)
- **背景**: 白色 (#ffffff)
- **文字**: 深灰 (#374151)

### Softmax阶段
- **渐变**: Viridis配色
  - 低概率: #440154 (深紫)
  - 中概率: #31688e (蓝绿)
  - 高概率: #fde724 (亮黄)

### 预测阶段
- **最高概率**: 金色系 (#fbbf24, #f59e0b)
- **其他**: Viridis配色（继承）
- **卡片背景**: 琥珀色渐变 (from-amber-50 to-yellow-50)

## D3.js关键技术

### 1. 数据绑定与更新

```typescript
const bars = g.selectAll('.logit-bar')
  .data(topKLogits)
  .join('rect')  // enter + update + exit 模式
  .attr('class', 'logit-bar')
  // ... 其他属性
```

### 2. 比例尺（Scales）

```typescript
// 分类比例尺（X轴）
const xScale = d3.scaleBand()
  .domain(logits.map(d => d.token))
  .range([0, width])
  .padding(0.2);

// 线性比例尺（Y轴）
const yScale = d3.scaleLinear()
  .domain([minLogit, maxLogit])
  .range([height, 0]);

// 颜色比例尺
const colorScale = d3.scaleSequential(d3.interpolateViridis)
  .domain([0, maxProbability]);
```

### 3. 过渡动画（Transitions）

```typescript
// 基础过渡
selection.transition()
  .duration(1000)
  .attr('y', newY)
  .attr('height', newHeight);

// 链式过渡
selection.transition()
  .duration(800)
  .attr('opacity', 1)
  .transition()  // 第二个过渡
  .duration(500)
  .attr('fill', 'gold');
```

### 4. 交互事件

```typescript
bars
  .on('mouseover', function(event, d) {
    d3.select(this)  // 选中当前元素
      .attr('opacity', 1)
      .attr('stroke', '#000');
    
    // 显示tooltip
    showTooltip(event, d);
  })
  .on('mouseout', function(event, d) {
    d3.select(this)
      .attr('opacity', 0.8)
      .attr('stroke', 'none');
    
    // 隐藏tooltip
    hideTooltip();
  });
```

### 5. 坐标轴

```typescript
// X轴
const xAxis = d3.axisBottom(xScale)
  .tickSize(0)
  .tickFormat(() => '');  // 不显示tick标签（使用自定义标签）

g.append('g')
  .attr('transform', `translate(0, ${height})`)
  .call(xAxis);

// Y轴
const yAxis = d3.axisLeft(yScale)
  .ticks(5)
  .tickFormat(d => `${(d * 100).toFixed(0)}%`);

g.append('g').call(yAxis);
```

## 性能优化

### 1. Top-K限制

只显示概率最高的K个token，而不是全部词汇表：

```typescript
logitDataArray.sort((a, b) => b.logit - a.logit);
const topK = logitDataArray.slice(0, 20);  // 只取前20
```

**原因**: 
- 词汇表通常有50,000+个token
- 渲染全部会导致性能问题
- 用户只关心最相关的候选

### 2. 虚拟化滚动（未来）

对于更大的K值，考虑使用虚拟滚动：
- 只渲染可见区域的元素
- 滚动时动态更新DOM

### 3. 缓存计算结果

```typescript
const memoizedLogits = useMemo(() => {
  // 计算logits
  return computeLogits(selectedVector, vocabulary);
}, [selectedVector, vocabulary]);
```

## 使用示例

### 基础用法

```typescript
import { OutputLayerViz } from '@/components/visualizations';

function MyComponent() {
  const finalHiddenState = [
    [0.1, 0.2, ...],  // token 0
    [0.3, 0.4, ...],  // token 1
    [0.5, 0.6, ...],  // token 2 (用于预测)
  ];
  
  const vocabulary = ['hello', 'world', 'the', ...];
  
  return (
    <OutputLayerViz
      finalHiddenState={finalHiddenState}
      vocabulary={vocabulary}
      onComplete={() => console.log('完成！')}
    />
  );
}
```

### 配合Demo使用

```typescript
import { OutputLayerDemo } from '@/components/visualizations';

function DemoPage() {
  return <OutputLayerDemo />;
}
```

## 集成到主可视化流程

### 在VisualizationCanvas中

```typescript
// 检测是否到达最后一步
if (currentStep === totalSteps - 1) {
  // 显示输出层
  return (
    <OutputLayerViz
      finalHiddenState={currentStepData.output_data}
      vocabulary={vocabularyFromBackend}
      onComplete={handleVisualizationComplete}
    />
  );
}
```

### 在解释面板中同步

```typescript
// ExplanationPanel.tsx中添加输出层步骤的说明
const explanations = {
  'final_layer_norm': '...',
  'token_selection': '...',
  'logits_head': '...',
  'softmax': '...',
  'prediction': '...',
};
```

## 自定义与扩展

### 1. 修改Top-K数量

```typescript
const topK = logitDataArray.slice(0, K);  // 修改K值
```

### 2. 添加温度参数

```typescript
interface OutputLayerVizProps {
  temperature?: number;  // 默认1.0
}

// 在Softmax计算中
const scaledLogits = logits.map(l => l / temperature);
```

### 3. 支持不同的解码策略

```typescript
// 贪婪解码
const prediction = predictions[0];

// Top-K采样
const topK = predictions.slice(0, K);
const sampled = sampleFromDistribution(topK);

// Top-P (Nucleus)采样
const topP = filterByProbability(predictions, P);
const sampled = sampleFromDistribution(topP);
```

### 4. 添加Beam Search可视化

显示多个候选序列的概率演化。

## 常见问题

### Q1: 为什么只显示Top-20？

A: 完整词汇表通常有数万个token，全部显示会导致：
- 性能问题（渲染慢）
- 可读性差（柱状图密集）
- 大部分token概率极低（不相关）

### Q2: Logits和概率有什么区别？

A: 
- **Logits**: 未归一化的原始分数，可以是任意实数
- **Probabilities**: 经过Softmax后的值，范围[0,1]，总和为1

### Q3: 如何理解负的Logit值？

A: Logit可以是负数，负值表示模型认为该token不太可能出现。经过exp()和Softmax后，负logit仍会得到正的概率，只是值较小。

### Q4: 为什么使用最后一个token？

A: 对于语言建模（Next Token Prediction），使用最后一个token的表示来预测序列中的下一个词，这是自回归语言模型（如GPT）的标准做法。

### Q5: 如何改变token选择策略？

A: 修改`TokenSelectionViz`组件，根据任务类型选择不同的token索引：
```typescript
const selectedIndex = taskType === 'classification' ? 0 : hiddenState.length - 1;
```

## 参考资料

- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [D3.js Documentation](https://d3js.org/)
- [Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

## 总结

输出层可视化组件提供了从Transformer隐藏状态到最终预测的完整视图，通过：

1. ✅ 清晰的阶段划分
2. ✅ 流畅的动画过渡
3. ✅ 丰富的交互功能
4. ✅ 详细的数据展示
5. ✅ 教育友好的设计

帮助用户深入理解大语言模型的预测机制。
