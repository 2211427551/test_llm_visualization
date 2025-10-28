# Tokenization & Embedding Visualization Components

这个目录包含了使用 D3.js 实现的 Transformer 词元化和嵌入过程的可视化组件。

## 组件概览

### 1. TokenizationViz

词元化可视化组件，展示文本分割成词元的动画过程。

**功能特性:**
- 文本平滑分割成词元方块
- 词元 ID 显示
- 交互式悬停提示（显示词元、ID、位置）
- 流畅的 SVG 动画
- 自适应宽度的词元方块

**使用示例:**
```tsx
import { TokenizationViz } from '@/components/visualizations';

<TokenizationViz
  text="hello world"
  tokens={[31373, 995]}
  tokenTexts={["hello", "world"]}
  onComplete={() => console.log('Tokenization complete')}
/>
```

**Props:**
- `text: string` - 原始输入文本
- `tokens: number[]` - 词元 ID 数组
- `tokenTexts: string[]` - 词元文本数组
- `onComplete?: () => void` - 动画完成回调

### 2. EmbeddingViz

嵌入查表与位置编码可视化组件，展示向量的生成和相加过程。

**功能特性:**
- 嵌入矩阵抽象表示
- 词元到嵌入向量的查找动画
- 位置编码向量生成
- 向量逐元素相加动画
- 颜色映射（RdBu 色阶）
- 交互式悬停提示（显示维度索引和数值）
- 支持串行/并行动画模式

**使用示例:**
```tsx
import { EmbeddingViz } from '@/components/visualizations';

<EmbeddingViz
  tokens={[31373, 995]}
  tokenTexts={["hello", "world"]}
  embeddings={embeddings}  // shape: [n_tokens, n_embd]
  positionalEncodings={positionalEncodings}  // shape: [n_tokens, n_embd]
  nEmbd={768}
  nVocab={50257}
  animationMode="serial"
  onComplete={() => console.log('Embedding complete')}
/>
```

**Props:**
- `tokens: number[]` - 词元 ID 数组
- `tokenTexts: string[]` - 词元文本数组
- `embeddings: number[][]` - 嵌入向量矩阵 [n_tokens, n_embd]
- `positionalEncodings: number[][]` - 位置编码矩阵 [n_tokens, n_embd]
- `nEmbd: number` - 嵌入维度大小
- `nVocab: number` - 词汇表大小
- `animationMode?: 'serial' | 'parallel'` - 动画模式（默认: 'serial'）
- `onComplete?: () => void` - 动画完成回调

### 3. TokenEmbeddingVisualization

集成组件，依次展示词元化和嵌入过程。

**功能特性:**
- 自动阶段切换（词元化 → 嵌入 → 完成）
- 动画模式切换（串行/并行）
- 重新播放功能
- 进度指示
- 完成状态展示

**使用示例:**
```tsx
import { TokenEmbeddingVisualization } from '@/components/visualizations';

<TokenEmbeddingVisualization
  text="hello world"
  tokens={[31373, 995]}
  tokenTexts={["hello", "world"]}
  embeddings={embeddings}
  positionalEncodings={positionalEncodings}
  nEmbd={768}
  nVocab={50257}
/>
```

**Props:**
- 组合了 `TokenizationViz` 和 `EmbeddingViz` 的所有 props

### 4. TokenEmbeddingDemo

完整的演示组件，包含输入控制和说明文档。

**功能特性:**
- 文本输入控制
- 配置信息显示
- 自动生成模拟数据
- 说明文档
- 完整的用户界面

**使用示例:**
```tsx
import { TokenEmbeddingDemo } from '@/components/visualizations';

<TokenEmbeddingDemo />
```

## 技术实现

### D3.js v7

使用 D3.js 的核心功能：
- **选择器 (Selections)**: 数据绑定和 DOM 操作
- **过渡 (Transitions)**: 流畅的动画效果
- **缓动函数 (Easing)**: 自然的动画曲线
- **色阶 (Scales)**: 数值到颜色的映射

### 动画时序

```
TokenizationViz:
  500ms - 初始显示
  800ms - 文本淡出
  600ms - 词元方块出现（每个延迟 100ms）
  400ms - Token ID 显示（每个延迟 50ms）

EmbeddingViz (每个词元):
  600ms - 查找箭头动画
  400ms - 矩阵高亮
  600ms - 嵌入向量出现
  700ms - 位置编码出现
  400ms - 加号显示
  800ms - 向量合并
  600ms - 最终向量显示
```

### 颜色方案

- **词元方块**: Blues 插值 (d3.interpolateBlues)
- **嵌入向量**: RdBu 发散色阶 (d3.interpolateRdBu)
- **位置编码**: Greens 插值 (d3.interpolateGreens)
- **最终向量**: RdBu 发散色阶

### 响应式设计

- SVG viewBox 自适应容器宽度
- 动态计算方块大小和间距
- 根据维度数量自动调整显示数量
- 虚拟化渲染（仅显示可见的维度）

## 性能优化

1. **虚拟滚动**: 对于大维度（n_embd=768），只渲染可见的方块
2. **数据采样**: 自动计算合适的显示维度数量
3. **动画队列**: 使用 setTimeout 和 transition.on('end') 控制动画流程
4. **内存清理**: useEffect 返回清理函数，移除所有 SVG 元素

## 交互功能

### 悬停提示 (Tooltips)

所有可视化元素都支持悬停交互：
- **词元方块**: 显示词元文本、ID、位置
- **嵌入向量方块**: 显示维度索引和数值
- **位置编码方块**: 显示维度索引和数值
- **最终向量方块**: 显示合并信息和数值

### 高亮效果

鼠标悬停时：
- 方块描边加粗
- 颜色变亮
- 显示详细信息浮层

## 集成到项目

### 在 VisualizationCanvas 中使用

```tsx
import { TokenEmbeddingVisualization } from './visualizations';

// 在组件中
<TokenEmbeddingVisualization
  text={inputText}
  tokens={tokens}
  tokenTexts={tokenTexts}
  embeddings={embeddings}
  positionalEncodings={positionalEncodings}
  nEmbd={config.n_embd}
  nVocab={config.n_vocab}
/>
```

### 生成位置编码

```typescript
const generatePositionalEncodings = (numTokens: number, nEmbd: number) => {
  return Array.from({ length: numTokens }, (_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );
};
```

## 样式定制

组件使用 Tailwind CSS 类和内联 D3 样式。可以通过以下方式定制：

1. **修改颜色方案**: 更改 D3 插值函数
2. **调整动画速度**: 修改 transition.duration() 值
3. **改变布局**: 调整坐标常量（tokenY, embeddingVectorY 等）
4. **自定义缓动**: 使用不同的 d3.ease* 函数

## 浏览器兼容性

- 现代浏览器 (Chrome, Firefox, Safari, Edge)
- 需要 SVG 2 支持
- 需要 ES6+ 支持

## 未来改进

- [ ] 添加动画暂停/恢复控制
- [ ] 支持更多的嵌入矩阵可视化模式
- [ ] 添加音效反馈
- [ ] 导出动画为 GIF/视频
- [ ] 添加可访问性 (ARIA) 支持
- [ ] 性能监控和优化
