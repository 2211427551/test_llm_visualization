export interface Token {
  text: string;
  id: number;
}

export interface LayerData {
  layerId: number;
  layerName: string;
  inputShape: number[];
  outputShape: number[];
  activations?: number[][];
  weights?: number[][];
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
