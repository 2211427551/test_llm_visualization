export type LayerType = 'input' | 'embedding' | 'attention' | 'feedforward' | 'output'

type NumericMatrix = number[][]

export interface SparseAttentionCell {
  value: number
  isSparse: boolean
}

export interface SparseAttentionData {
  headLabels: string[]
  matrix: SparseAttentionCell[][]
  note: string
}

export interface MoERouteDefinition {
  tokenId: string
  expertId: string
  weight: number
}

export interface MoERoutingData {
  tokens: { id: string; label: string }[]
  experts: { id: string; label: string }[]
  routes: MoERouteDefinition[]
  description: string
}

export interface LayerVisualizationData {
  id: string
  name: string
  type: LayerType
  summary: string
  tensorHeatmap: NumericMatrix
  sparseAttention?: SparseAttentionData
  moeRouting?: MoERoutingData
}

export interface StepVisualizationData {
  id: string
  name: string
  description: string
  layers: LayerVisualizationData[]
}
