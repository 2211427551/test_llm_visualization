import { create } from 'zustand';
import { ExecutionStore, ModelForwardResponse } from '../types';

export const useExecutionStore = create<ExecutionStore>((set, get) => ({
  status: 'idle',
  currentStepIndex: 0,
  data: null,
  error: null,
  isPlaying: false,
  playbackSpeed: 1,
  selectedLayerId: null,
  breadcrumbs: [],

  setStatus: (status) => set({ status }),

  setCurrentStep: (index) => {
    const { data } = get();
    if (data && index >= 0 && index < data.steps.length) {
      set({ currentStepIndex: index });
    }
  },

  setData: (data: ModelForwardResponse) => set({ 
    data, 
    currentStepIndex: 0,
    status: 'success',
    error: null 
  }),

  setError: (error) => set({ error, status: 'error' }),

  setIsPlaying: (isPlaying) => set({ isPlaying }),

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  setSelectedLayer: (layerId) => set({ selectedLayerId: layerId }),

  setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs }),

  nextStep: () => {
    const { data, currentStepIndex } = get();
    if (data && currentStepIndex < data.steps.length - 1) {
      set({ currentStepIndex: currentStepIndex + 1 });
    } else {
      set({ isPlaying: false });
    }
  },

  previousStep: () => {
    const { currentStepIndex } = get();
    if (currentStepIndex > 0) {
      set({ currentStepIndex: currentStepIndex - 1 });
    }
  },

  reset: () => set({
    status: 'idle',
    currentStepIndex: 0,
    data: null,
    error: null,
    isPlaying: false,
    selectedLayerId: null,
    breadcrumbs: [],
  }),
}));
