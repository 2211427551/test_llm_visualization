export interface Token {
  text: string;
  id: number;
}

export interface AttentionData {
  queryMatrix?: number[][];
  keyMatrix?: number[][];
  valueMatrix?: number[][];
  attentionScores?: number[][];
  sparsityMask?: number[][];
  numHeads?: number;
  headDim?: number;
}

export interface ExpertData {
  expertId: number;
  activations: number[];
}

export interface MoEData {
  gatingWeights?: number[][];
  selectedExperts?: number[][];
  expertActivations?: ExpertData[];
  numExperts?: number;
  topK?: number;
}

export type LayerType = 'attention' | 'moe' | 'feedforward' | 'embedding' | 'normalization' | 'output';

export interface LayerData {
  layerId: number;
  layerName: string;
  layerType?: LayerType;
  inputShape: number[];
  outputShape: number[];
  activations?: number[][];
  weights?: number[][];
  attentionData?: AttentionData;
  moeData?: MoEData;
  truncated?: boolean;
}

export interface ComputationStep {
  stepIndex: number;
  layerData: LayerData;
  description: string;
}

export interface ModelForwardResponse {
  success: boolean;
  inputText: string;
  tokens: Token[];
  tokenCount: number;
  steps: ComputationStep[];
  outputProbabilities: {
    token: string;
    probability: number;
  }[];
  warnings?: string[];
  truncated?: boolean;
}

export interface RunState {
  status: 'idle' | 'loading' | 'success' | 'error';
  currentStepIndex: number;
  data: ModelForwardResponse | null;
  error: string | null;
  isPlaying: boolean;
  playbackSpeed: number;
}

export interface ExecutionStore extends RunState {
  setStatus: (status: RunState['status']) => void;
  setCurrentStep: (index: number) => void;
  setData: (data: ModelForwardResponse) => void;
  setError: (error: string | null) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  nextStep: () => void;
  previousStep: () => void;
  reset: () => void;
}
