# D3.js Tokenization & Embedding Visualization Guide

本指南介绍如何使用新创建的 D3.js 可视化组件来展示 Transformer 的词元化和嵌入过程。

## 快速开始

### 查看演示

启动开发服务器后访问演示页面：

```bash
cd frontend
npm run dev
# 访问 http://localhost:3000/demo
```

## 组件使用说明

### 1. 独立演示组件 (推荐用于测试)

最简单的使用方式是使用 `TokenEmbeddingDemo` 组件：

```tsx
import { TokenEmbeddingDemo } from '@/components/visualizations';

export default function Page() {
  return <TokenEmbeddingDemo />;
}
```

这个组件包含：
- 输入文本控制
- 配置信息显示
- 自动数据生成
- 完整的动画流程
- 重新播放按钮

### 2. 集成到现有应用

在 `VisualizationCanvas` 中已经集成了可视化功能。用户可以通过按钮切换：
- **D3.js 动画**: 显示交互式可视化
- **数据视图**: 显示 JSON 数据

### 3. 单独使用各个组件

#### TokenizationViz - 词元化可视化

```tsx
import { TokenizationViz } from '@/components/visualizations';

function MyComponent() {
  return (
    <TokenizationViz
      text="hello world"
      tokens={[31373, 995]}
      tokenTexts={["hello", "world"]}
      onComplete={() => console.log('Tokenization complete')}
    />
  );
}
```

#### EmbeddingViz - 嵌入与位置编码可视化

```tsx
import { EmbeddingViz } from '@/components/visualizations';

function MyComponent() {
  const embeddings = generateEmbeddings(); // [n_tokens, n_embd]
  const positionalEncodings = generatePositionalEncodings(); // [n_tokens, n_embd]

  return (
    <EmbeddingViz
      tokens={[31373, 995]}
      tokenTexts={["hello", "world"]}
      embeddings={embeddings}
      positionalEncodings={positionalEncodings}
      nEmbd={768}
      nVocab={50257}
      animationMode="serial" // 或 "parallel"
      onComplete={() => console.log('Embedding complete')}
    />
  );
}
```

## 数据准备

### 生成嵌入向量 (示例)

```typescript
// 生成随机嵌入向量（用于演示）
const generateEmbeddings = (tokens: number[], nEmbd: number) => {
  return tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );
};
```

### 生成位置编码

```typescript
// 标准的正弦/余弦位置编码
const generatePositionalEncodings = (numTokens: number, nEmbd: number) => {
  return Array.from({ length: numTokens }, (_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );
};
```

### 从后端 API 获取真实数据

```typescript
// 使用 InitResponse 中的 initial_state
const response = await initComputation(inputText, config);
const { embeddings, positional_encodings } = response.initial_state;

<TokenEmbeddingVisualization
  text={inputText}
  tokens={response.tokens}
  tokenTexts={response.token_texts}
  embeddings={embeddings}
  positionalEncodings={positional_encodings}
  nEmbd={config.n_embd}
  nVocab={config.n_vocab}
/>
```

## 可视化特性

### 词元化阶段

1. **初始显示**: 完整文本居中显示
2. **分割动画**: 文本淡出，词元方块从原位置出现
3. **排列**: 词元方块向下移动，水平排列
4. **ID 显示**: 每个词元下方显示其 token ID
5. **交互**: 悬停显示详细信息（词元、ID、位置）

### 嵌入阶段

1. **词元显示**: 顶部显示所有词元方块
2. **矩阵表示**: 中间显示嵌入矩阵的抽象表示 (n_vocab × n_embd)
3. **查表动画**: 箭头指向矩阵，矩阵闪烁
4. **向量提取**: 嵌入向量以彩色方块序列形式出现
5. **位置编码**: 同时生成位置编码向量（绿色系）
6. **向量相加**: 大的"+"号，两个向量合并
7. **最终向量**: 显示合并后的输入向量
8. **交互**: 悬停显示每个维度的索引和数值

### 动画模式

- **串行模式 (Serial)**: 逐个处理词元，适合教学演示
- **并行模式 (Parallel)**: 同时处理所有词元，更真实的并行计算

## 自定义配置

### 修改动画速度

在组件文件中找到 `.transition().duration()` 调用，修改毫秒值：

```typescript
// TokenizationViz.tsx
.transition()
  .duration(800)  // 改为 400 会加快一倍

// EmbeddingViz.tsx
.transition()
  .duration(600)  // 调整查表动画速度
```

### 修改颜色方案

```typescript
// 词元方块颜色
.attr('fill', d3.interpolateBlues(0.5 + d.index * 0.1 / tokenTexts.length))

// 嵌入向量颜色
const colorScale = d3.scaleSequential(d3.interpolateRdBu)
  .domain([1, -1]);

// 位置编码颜色
.attr('fill', d => d3.interpolateGreens(Math.abs(d)))
```

可用的 D3 颜色插值函数：
- `d3.interpolateRdBu` - 红蓝发散
- `d3.interpolateViridis` - 青黄渐变
- `d3.interpolateInferno` - 暗红橙黄
- `d3.interpolateBlues` - 蓝色渐变
- `d3.interpolateGreens` - 绿色渐变

### 修改布局位置

在组件中调整 Y 坐标常量：

```typescript
// EmbeddingViz.tsx
const tokenY = 50;                  // 词元位置
const matrixY = 150;                // 矩阵位置
const embeddingVectorY = 380;       // 嵌入向量位置
const positionalVectorY = 480;      // 位置编码位置
const additionY = 580;              // 加号位置
const finalVectorY = 680;           // 最终向量位置
```

## 响应式设计

组件自动适应容器宽度：

```typescript
const container = containerRef.current;
const width = container.clientWidth;  // 动态获取宽度
```

建议在不同屏幕尺寸测试：
- **大屏 (>1200px)**: 完整显示所有维度
- **中屏 (768-1200px)**: 显示部分维度
- **小屏 (<768px)**: 显示最少维度，保持清晰度

## 性能考虑

### 大规模数据

对于大嵌入维度（如 GPT-3 的 12288），组件会自动：

```typescript
const displayEmbd = Math.min(nEmbd, Math.floor((width - 200) / blockSize));
```

只显示可见的维度数量，避免渲染成千上万的 SVG 元素。

### 多词元处理

- **串行模式**: 内存友好，但动画较长
- **并行模式**: 更快完成，但对于多词元会占用更多 DOM 元素

建议：
- ≤5 个词元：使用并行模式
- >5 个词元：使用串行模式

## 常见问题

### Q: 为什么看不到动画？

A: 检查：
1. D3.js 是否正确安装：`npm list d3`
2. 容器是否有足够的高度
3. 浏览器控制台是否有错误

### Q: 如何调整动画速度？

A: 在 VisualizationCanvas 中还没有全局速度控制。可以：
1. 直接修改组件文件中的 `duration()` 值
2. 或添加一个速度倍数 prop，乘以所有 duration

示例：
```typescript
interface Props {
  speedMultiplier?: number; // 默认 1.0
}

.transition()
  .duration(800 * (speedMultiplier || 1))
```

### Q: 可以导出动画吗？

A: 目前不支持。未来可以考虑：
- 使用 html2canvas 截图每一帧
- 使用 FFMPEG 合成视频
- 或使用 SVG 动画录制工具

### Q: 如何集成到 Zustand store？

A: 在 store 中添加可视化控制状态：

```typescript
// visualizationStore.ts
interface VisualizationState {
  // ... 现有状态
  vizAnimationSpeed: number;
  vizAnimationMode: 'serial' | 'parallel';
  showD3Viz: boolean;
  
  setVizAnimationSpeed: (speed: number) => void;
  setVizAnimationMode: (mode: 'serial' | 'parallel') => void;
  toggleD3Viz: () => void;
}
```

## 扩展功能

### 添加音效

```typescript
const playSound = (type: 'tokenize' | 'embed' | 'add') => {
  const audio = new Audio(`/sounds/${type}.mp3`);
  audio.play();
};

// 在动画关键点调用
.on('end', () => {
  playSound('embed');
  // ...
});
```

### 添加暂停/恢复

```typescript
const [isPaused, setIsPaused] = useState(false);

// 使用 D3 的 selection.interrupt() 暂停所有过渡
const pauseAnimation = () => {
  d3.selectAll('*').interrupt();
};
```

### 导出为静态图

```typescript
import html2canvas from 'html2canvas';

const exportAsPNG = async () => {
  const canvas = await html2canvas(svgRef.current);
  const link = document.createElement('a');
  link.download = 'visualization.png';
  link.href = canvas.toDataURL();
  link.click();
};
```

## 参考资源

- [D3.js 官方文档](https://d3js.org/)
- [D3 Transitions](https://d3js.org/d3-transition)
- [D3 Color Schemes](https://d3js.org/d3-scale-chromatic)
- [SVG 教程](https://developer.mozilla.org/en-US/docs/Web/SVG)

## 维护者

如需修改或扩展可视化功能，请参考：
- `/frontend/src/components/visualizations/README.md` - 组件详细文档
- `/frontend/src/components/visualizations/` - 组件源代码
