import type {
  LayerVisualizationData,
  MoERouteDefinition,
  SparseAttentionCell,
  StepVisualizationData,
} from '../types/visualization'

type MatrixGenerator = (rowIndex: number, columnIndex: number) => number

const clamp = (value: number) => Math.max(0, Math.min(1, value))

const createHeatmapMatrix = (rows: number, cols: number, generator: MatrixGenerator): number[][] =>
  Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => Number(clamp(generator(row, col)).toFixed(3))),
  )

const createSparseAttentionMatrix = (
  size: number,
  emphasisColumns: number[],
  sparsityBias = 0.12,
): SparseAttentionCell[][] => {
  const emphasisSet = new Set(emphasisColumns)

  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) => {
      const distanceFactor = Math.exp(-Math.abs(row - col) / 1.8)
      const emphasisBoost = emphasisSet.has(col) ? 0.38 : 0
      const base = clamp(distanceFactor * 0.35 + emphasisBoost)
      const value = row === col ? 0.82 : base
      const isSparse = value < sparsityBias

      return {
        value: Number(value.toFixed(3)),
        isSparse,
      }
    }),
  )
}

const createMoERoutes = (
  tokenWeights: number[][],
  expertIds: string[],
): MoERouteDefinition[] => {
  return tokenWeights.flatMap((weights, tokenIndex) =>
    weights.map((weight, expertIndex) => ({
      tokenId: `token-${tokenIndex + 1}`,
      expertId: expertIds[expertIndex],
      weight: Number(weight.toFixed(3)),
    })),
  )
}

const visualizationSteps: StepVisualizationData[] = [
  {
    id: 'step-1',
    name: '输入解析',
    description: '对原始文本进行分词、嵌入和位置编码，为后续 Transformer 层准备上下文表示。',
    layers: [
      {
        id: 'layer-1-input',
        name: '分词嵌入',
        type: 'input',
        summary: '将汉字序列映射为稠密的向量表示，保留语义距离。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => Math.sin((i + 1) * 0.6) * 0.25 + Math.cos((j + 1) * 0.4) * 0.25 + 0.5),
        sparseAttention: {
          headLabels: Array.from({ length: 8 }, (_, index) => `位置 ${index + 1}`),
          matrix: createSparseAttentionMatrix(8, [1, 2, 5], 0.1),
          note: '输入阶段的注意力更集中在局部窗口，用于捕捉短程依赖。',
        },
        moeRouting: {
          tokens: Array.from({ length: 4 }, (_, index) => ({ id: `token-${index + 1}`, label: `令牌 ${index + 1}` })),
          experts: [
            { id: 'expert-a', label: '专家 A' },
            { id: 'expert-b', label: '专家 B' },
            { id: 'expert-c', label: '专家 C' },
          ],
          routes: createMoERoutes(
            [
              [0.55, 0.35, 0.1],
              [0.62, 0.28, 0.1],
              [0.28, 0.5, 0.22],
              [0.2, 0.58, 0.22],
            ],
            ['expert-a', 'expert-b', 'expert-c'],
          ),
          description: '早期令牌主要被路由到语言先验更强的专家 A 与模式识别专家 B。',
        },
      },
      {
        id: 'layer-1-position',
        name: '位置编码',
        type: 'embedding',
        summary: '利用正弦余弦基函数构造位置特征，与词向量融合。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => 0.5 + Math.sin((i + j) * 0.4) * 0.4),
      },
      {
        id: 'layer-1-norm',
        name: '归一化融合',
        type: 'embedding',
        summary: '通过层归一化稳定分布，避免梯度不稳定。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => 0.48 + Math.sin(i * 0.3) * 0.15 + Math.cos(j * 0.25) * 0.15),
      },
    ],
  },
  {
    id: 'step-2',
    name: '注意力推理',
    description: '多头注意力捕捉长程依赖，同时稀疏化策略减少冗余计算。',
    layers: [
      {
        id: 'layer-2-attention',
        name: '多头注意力',
        type: 'attention',
        summary: '聚合上下文信息，重点关注动宾结构与关键实体。',
        tensorHeatmap: createHeatmapMatrix(10, 10, (i, j) => 0.55 + 0.25 * Math.cos((i - j) * 0.45)),
        sparseAttention: {
          headLabels: Array.from({ length: 10 }, (_, index) => `位置 ${index + 1}`),
          matrix: createSparseAttentionMatrix(10, [0, 4, 8], 0.08),
          note: '稀疏头聚焦到关键主谓语，非稀疏区域色彩更深。',
        },
        moeRouting: {
          tokens: Array.from({ length: 5 }, (_, index) => ({ id: `token-${index + 1}`, label: `令牌 ${index + 1}` })),
          experts: [
            { id: 'expert-a', label: '专家 A' },
            { id: 'expert-b', label: '专家 B' },
            { id: 'expert-d', label: '专家 D' },
          ],
          routes: createMoERoutes(
            [
              [0.38, 0.49, 0.13],
              [0.26, 0.61, 0.13],
              [0.18, 0.65, 0.17],
              [0.12, 0.7, 0.18],
              [0.22, 0.5, 0.28],
            ],
            ['expert-a', 'expert-b', 'expert-d'],
          ),
          description: '推理阶段更多令牌被派发到语义抽象能力更强的专家 B。',
        },
      },
      {
        id: 'layer-2-sparse-attn',
        name: '稀疏注意力头',
        type: 'attention',
        summary: '稀疏模式仅保留高贡献连接，大幅降低复杂度。',
        tensorHeatmap: createHeatmapMatrix(10, 10, (i, j) => 0.35 + 0.45 * Math.exp(-Math.abs(i - j) * 0.35)),
        sparseAttention: {
          headLabels: Array.from({ length: 10 }, (_, index) => `位置 ${index + 1}`),
          matrix: createSparseAttentionMatrix(10, [1, 3, 7], 0.05),
          note: '亮色方块表示被保留的高权重连接，灰色为稀疏区域。',
        },
      },
      {
        id: 'layer-2-ffn',
        name: '前馈网络',
        type: 'feedforward',
        summary: '逐位置非线性变换，放大关键信号。',
        tensorHeatmap: createHeatmapMatrix(10, 10, (i, j) => 0.42 + Math.sin(i * 0.5) * 0.28 + Math.cos(j * 0.32) * 0.18),
      },
    ],
  },
  {
    id: 'step-3',
    name: '混合专家聚合',
    description: '门控网络整合不同专家输出，完成最终生成与预测。',
    layers: [
      {
        id: 'layer-3-gating',
        name: '门控决策',
        type: 'feedforward',
        summary: '结合注意力上下文与状态，动态选择需要激活的专家。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => 0.48 + 0.35 * Math.exp(-Math.abs(i - j) * 0.4)),
        moeRouting: {
          tokens: Array.from({ length: 6 }, (_, index) => ({ id: `token-${index + 1}`, label: `令牌 ${index + 1}` })),
          experts: [
            { id: 'expert-b', label: '专家 B' },
            { id: 'expert-c', label: '专家 C' },
            { id: 'expert-e', label: '专家 E' },
            { id: 'expert-f', label: '专家 F' },
          ],
          routes: createMoERoutes(
            [
              [0.12, 0.62, 0.16, 0.1],
              [0.1, 0.65, 0.18, 0.07],
              [0.22, 0.5, 0.18, 0.1],
              [0.2, 0.38, 0.24, 0.18],
              [0.18, 0.34, 0.28, 0.2],
              [0.15, 0.32, 0.3, 0.23],
            ],
            ['expert-b', 'expert-c', 'expert-e', 'expert-f'],
          ),
          description: '输出阶段逐步引入风格专家 E 与语气调节专家 F，保证表达自然。',
        },
      },
      {
        id: 'layer-3-combine',
        name: '专家聚合',
        type: 'feedforward',
        summary: '将多专家输出按照门控权重合并为最终表征。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => 0.52 + 0.3 * Math.sin((i + j) * 0.35)),
      },
      {
        id: 'layer-3-output',
        name: '输出投影',
        type: 'output',
        summary: '映射至词表空间，用于生成最终预测结果。',
        tensorHeatmap: createHeatmapMatrix(8, 8, (i, j) => 0.56 + 0.25 * Math.cos(i * 0.4) * Math.sin(j * 0.45)),
      },
    ],
  },
]

export default visualizationSteps
