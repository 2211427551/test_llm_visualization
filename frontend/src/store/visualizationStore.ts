import { create } from 'zustand';
import { ModelConfig, StepResponse } from '@/types';
import { initComputation, fetchStep } from '@/services/api';

interface VisualizationState {
  // Session information
  sessionId: string | null;
  
  // Computation state
  isInitialized: boolean;
  isPlaying: boolean;
  isLoading: boolean;
  currentStep: number;
  totalSteps: number;
  playbackSpeed: number;
  
  // Data
  inputText: string;
  tokens: number[];
  tokenTexts: string[];
  currentStepData: StepResponse | null;
  
  // Configuration
  config: ModelConfig;
  
  // Error handling
  error: string | null;
  
  // Actions
  setInputText: (text: string) => void;
  setConfig: (config: Partial<ModelConfig>) => void;
  initializeComputation: () => Promise<void>;
  nextStep: () => Promise<void>;
  prevStep: () => Promise<void>;
  goToStep: (step: number) => Promise<void>;
  togglePlayback: () => void;
  setPlaybackSpeed: (speed: number) => void;
  reset: () => void;
}

const DEFAULT_CONFIG: ModelConfig = {
  n_vocab: 50257,
  n_embd: 768,
  n_layer: 12,
  n_head: 12,
  d_k: 64,
  max_seq_len: 512,
};

export const useVisualizationStore = create<VisualizationState>((set, get) => ({
  // Initial state
  sessionId: null,
  isInitialized: false,
  isPlaying: false,
  isLoading: false,
  currentStep: 0,
  totalSteps: 0,
  playbackSpeed: 1.0,
  inputText: '',
  tokens: [],
  tokenTexts: [],
  currentStepData: null,
  config: DEFAULT_CONFIG,
  error: null,

  // Actions
  setInputText: (text: string) => {
    set({ inputText: text, error: null });
  },

  setConfig: (config: Partial<ModelConfig>) => {
    set((state) => ({
      config: { ...state.config, ...config },
    }));
  },

  initializeComputation: async () => {
    const { inputText, config } = get();
    
    if (!inputText.trim()) {
      set({ error: '请输入文本' });
      return;
    }

    set({ isLoading: true, error: null });

    try {
      const response = await initComputation(inputText, config);
      
      set({
        sessionId: response.session_id,
        tokens: response.tokens,
        tokenTexts: response.token_texts,
        totalSteps: response.total_steps,
        isInitialized: true,
        currentStep: 0,
        isLoading: false,
      });

      // Load first step
      if (response.total_steps > 0) {
        get().goToStep(0);
      }
    } catch (error) {
      const errorMessage = error instanceof Error && 'response' in error
        ? (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
        : undefined;
      set({
        error: errorMessage || '初始化失败，请检查后端服务',
        isLoading: false,
      });
    }
  },

  nextStep: async () => {
    const { currentStep, totalSteps, sessionId } = get();
    
    if (currentStep < totalSteps - 1 && sessionId) {
      await get().goToStep(currentStep + 1);
    }
  },

  prevStep: async () => {
    const { currentStep, sessionId } = get();
    
    if (currentStep > 0 && sessionId) {
      await get().goToStep(currentStep - 1);
    }
  },

  goToStep: async (step: number) => {
    const { sessionId, totalSteps } = get();
    
    if (!sessionId) {
      set({ error: '会话未初始化' });
      return;
    }

    if (step < 0 || step >= totalSteps) {
      set({ error: '步骤索引超出范围' });
      return;
    }

    set({ isLoading: true, error: null });

    try {
      const stepData = await fetchStep(sessionId, step);
      set({
        currentStep: step,
        currentStepData: stepData,
        isLoading: false,
      });
    } catch (error) {
      const errorMessage = error instanceof Error && 'response' in error
        ? (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
        : undefined;
      set({
        error: errorMessage || '获取步骤数据失败',
        isLoading: false,
      });
    }
  },

  togglePlayback: () => {
    set((state) => ({ isPlaying: !state.isPlaying }));
  },

  setPlaybackSpeed: (speed: number) => {
    set({ playbackSpeed: speed });
  },

  reset: () => {
    set({
      sessionId: null,
      isInitialized: false,
      isPlaying: false,
      isLoading: false,
      currentStep: 0,
      totalSteps: 0,
      playbackSpeed: 1.0,
      inputText: '',
      tokens: [],
      tokenTexts: [],
      currentStepData: null,
      error: null,
    });
  },
}));
