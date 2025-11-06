import type { StepVisualizationData } from './visualization'

export interface ModelConfig {
  modelName: string
  vocabSize: number
  contextSize: number
  nLayer: number
  nHead: number
  nEmbed: number
  dropout: number
  useSparseAttention: boolean
  useMoe: boolean
  moeNumExperts: number
  moeTopK: number
  initializedAt: string
  [key: string]: unknown
}

export interface VisualizationRuntimeSummary {
  capturedAt: string
  forwardTimeMs: number
  memoryMB: number
  gpuUtilization: number
  batchSize: number
  sequenceLength: number
  logitsShape: number[]
}

export interface CapturedVisualizationData {
  steps: StepVisualizationData[]
  runtime: VisualizationRuntimeSummary
  tokenSequence: string[]
  modelSummary: {
    nLayer: number
    nHead: number
    nEmbed: number
    useSparseAttention: boolean
    useMoe: boolean
  }
}

export interface InitializeResult {
  success: boolean
  message: string
  config: ModelConfig
}

export interface ForwardResult {
  success: boolean
  message: string
  logitsShape: number[]
  sequenceLength: number
  capturedData?: CapturedVisualizationData
}
