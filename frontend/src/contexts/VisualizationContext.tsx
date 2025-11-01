'use client';

import React, { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';

// Visualization state types
export interface VisualizationState {
  // Current view and navigation
  currentView: 'overview' | 'embedding' | 'attention' | 'moe' | 'output';
  currentStep: number;
  totalSteps: number;
  
  // Module states
  modules: {
    embedding: {
      isActive: boolean;
      tokens: string[];
      embeddings: number[][];
      positionalEncodings: number[][];
    };
    attention: {
      isActive: boolean;
      heads: number;
      attentionWeights: number[][] | number[][];
      queries: number[][];
      keys: number[][];
      values: number[][];
    };
    moe: {
      isActive: boolean;
      experts: number;
      expertWeights: number[];
      gatingScores: number[];
    };
    output: {
      isActive: boolean;
      logits: number[];
      probabilities: number[];
      predictions: { token: string; probability: number }[];
    };
  };
  
  // Visualization settings
  settings: {
    animationSpeed: number;
    showConnections: boolean;
    showValues: boolean;
    colorScheme: 'default' | 'heatmap' | 'categorical';
    viewMode: '2d' | '3d' | 'mixed';
  };
  
  // Interaction state
  interaction: {
    hoveredElement: string | null;
    selectedElement: string | null;
    isPlaying: boolean;
    isPaused: boolean;
  };
}

// Action types
export type VisualizationAction =
  | { type: 'SET_VIEW'; payload: VisualizationState['currentView'] }
  | { type: 'SET_STEP'; payload: number }
  | { type: 'SET_TOTAL_STEPS'; payload: number }
  | { type: 'UPDATE_MODULE'; payload: { module: keyof VisualizationState['modules']; data: Partial<VisualizationState['modules'][keyof VisualizationState['modules']]> } }
  | { type: 'UPDATE_SETTINGS'; payload: Partial<VisualizationState['settings']> }
  | { type: 'SET_INTERACTION'; payload: Partial<VisualizationState['interaction']> }
  | { type: 'RESET_VISUALIZATION' }
  | { type: 'LOAD_DATA'; payload: Partial<VisualizationState> };

// Initial state
const initialState: VisualizationState = {
  currentView: 'overview',
  currentStep: 0,
  totalSteps: 0,
  
  modules: {
    embedding: {
      isActive: false,
      tokens: [],
      embeddings: [],
      positionalEncodings: [],
    },
    attention: {
      isActive: false,
      heads: 8,
      attentionWeights: [],
      queries: [],
      keys: [],
      values: [],
    },
    moe: {
      isActive: false,
      experts: 4,
      expertWeights: [],
      gatingScores: [],
    },
    output: {
      isActive: false,
      logits: [],
      probabilities: [],
      predictions: [],
    },
  },
  
  settings: {
    animationSpeed: 1.0,
    showConnections: true,
    showValues: false,
    colorScheme: 'default',
    viewMode: 'mixed',
  },
  
  interaction: {
    hoveredElement: null,
    selectedElement: null,
    isPlaying: false,
    isPaused: false,
  },
};

// Reducer
function visualizationReducer(state: VisualizationState, action: VisualizationAction): VisualizationState {
  switch (action.type) {
    case 'SET_VIEW':
      return { ...state, currentView: action.payload };
    
    case 'SET_STEP':
      return { 
        ...state, 
        currentStep: Math.max(0, Math.min(action.payload, state.totalSteps - 1))
      };
    
    case 'SET_TOTAL_STEPS':
      return { ...state, totalSteps: action.payload };
    
    case 'UPDATE_MODULE':
      return {
        ...state,
        modules: {
          ...state.modules,
          [action.payload.module]: {
            ...state.modules[action.payload.module],
            ...action.payload.data,
          },
        },
      };
    
    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: {
          ...state.settings,
          ...action.payload,
        },
      };
    
    case 'SET_INTERACTION':
      return {
        ...state,
        interaction: {
          ...state.interaction,
          ...action.payload,
        },
      };
    
    case 'RESET_VISUALIZATION':
      return initialState;
    
    case 'LOAD_DATA':
      return {
        ...state,
        ...action.payload,
      };
    
    default:
      return state;
  }
}

// Context
const VisualizationContext = createContext<{
  state: VisualizationState;
  dispatch: React.Dispatch<VisualizationAction>;
  actions: {
    setView: (view: VisualizationState['currentView']) => void;
    setStep: (step: number) => void;
    nextStep: () => void;
    previousStep: () => void;
    updateModule: (module: keyof VisualizationState['modules'], data: Partial<VisualizationState['modules'][keyof VisualizationState['modules']]>) => void;
    updateSettings: (settings: Partial<VisualizationState['settings']>) => void;
    setHoveredElement: (element: string | null) => void;
    setSelectedElement: (element: string | null) => void;
    play: () => void;
    pause: () => void;
    reset: () => void;
    loadData: (data: Partial<VisualizationState>) => void;
  };
} | null>(null);

// Provider
export function VisualizationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(visualizationReducer, initialState);
  
  // Actions
  const actions = {
    setView: useCallback((view: VisualizationState['currentView']) => {
      dispatch({ type: 'SET_VIEW', payload: view });
    }, []),
    
    setStep: useCallback((step: number) => {
      dispatch({ type: 'SET_STEP', payload: step });
    }, []),
    
    nextStep: useCallback(() => {
      dispatch({ type: 'SET_STEP', payload: state.currentStep + 1 });
    }, [state.currentStep]),
    
    previousStep: useCallback(() => {
      dispatch({ type: 'SET_STEP', payload: state.currentStep - 1 });
    }, [state.currentStep]),
    
    updateModule: useCallback((
      module: keyof VisualizationState['modules'],
      data: Partial<VisualizationState['modules'][keyof VisualizationState['modules']]>
    ) => {
      dispatch({ type: 'UPDATE_MODULE', payload: { module, data } });
    }, []),
    
    updateSettings: useCallback((settings: Partial<VisualizationState['settings']>) => {
      dispatch({ type: 'UPDATE_SETTINGS', payload: settings });
    }, []),
    
    setHoveredElement: useCallback((element: string | null) => {
      dispatch({ type: 'SET_INTERACTION', payload: { hoveredElement: element } });
    }, []),
    
    setSelectedElement: useCallback((element: string | null) => {
      dispatch({ type: 'SET_INTERACTION', payload: { selectedElement: element } });
    }, []),
    
    play: useCallback(() => {
      dispatch({ type: 'SET_INTERACTION', payload: { isPlaying: true, isPaused: false } });
    }, []),
    
    pause: useCallback(() => {
      dispatch({ type: 'SET_INTERACTION', payload: { isPlaying: false, isPaused: true } });
    }, []),
    
    reset: useCallback(() => {
      dispatch({ type: 'RESET_VISUALIZATION' });
    }, []),
    
    loadData: useCallback((data: Partial<VisualizationState>) => {
      dispatch({ type: 'LOAD_DATA', payload: data });
    }, []),
  };
  
  return (
    <VisualizationContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </VisualizationContext.Provider>
  );
}

// Hook
export function useVisualization() {
  const context = useContext(VisualizationContext);
  if (!context) {
    throw new Error('useVisualization must be used within a VisualizationProvider');
  }
  return context;
}