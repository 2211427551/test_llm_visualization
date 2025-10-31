'use client';

import React, { createContext, useContext, useReducer, useCallback, ReactNode, useRef } from 'react';
import { gsap } from 'gsap';

// Animation state types
export interface AnimationState {
  // Timeline state
  isPlaying: boolean;
  isPaused: boolean;
  currentTime: number;
  duration: number;
  progress: number;
  
  // Playback controls
  playbackSpeed: number;
  isLooping: boolean;
  isReversed: boolean;
  
  // Keyframes and breakpoints
  keyframes: Array<{
    id: string;
    time: number;
    label: string;
    description?: string;
  }>;
  
  // Current active animations
  activeAnimations: Array<{
    id: string;
    target: string;
    type: string;
    startTime: number;
    duration: number;
  }>;
  
  // Settings
  settings: {
    autoPlay: boolean;
    showKeyframes: boolean;
    smoothTransitions: boolean;
    easing: string;
  };
}

// Action types
export type AnimationAction =
  | { type: 'PLAY' }
  | { type: 'PAUSE' }
  | { type: 'STOP' }
  | { type: 'SEEK'; payload: number }
  | { type: 'SET_DURATION'; payload: number }
  | { type: 'SET_PLAYBACK_SPEED'; payload: number }
  | { type: 'TOGGLE_LOOP' }
  | { type: 'TOGGLE_REVERSE' }
  | { type: 'ADD_KEYFRAME'; payload: Omit<AnimationState['keyframes'][0], 'id'> }
  | { type: 'REMOVE_KEYFRAME'; payload: string }
  | { type: 'UPDATE_KEYFRAME'; payload: { id: string; updates: Partial<AnimationState['keyframes'][0]> } }
  | { type: 'ADD_ANIMATION'; payload: Omit<AnimationState['activeAnimations'][0], 'id'> }
  | { type: 'REMOVE_ANIMATION'; payload: string }
  | { type: 'UPDATE_SETTINGS'; payload: Partial<AnimationState['settings']> }
  | { type: 'RESET' };

// Initial state
const initialState: AnimationState = {
  isPlaying: false,
  isPaused: false,
  currentTime: 0,
  duration: 10, // 10 seconds default
  progress: 0,
  
  playbackSpeed: 1.0,
  isLooping: false,
  isReversed: false,
  
  keyframes: [],
  activeAnimations: [],
  
  settings: {
    autoPlay: false,
    showKeyframes: true,
    smoothTransitions: true,
    easing: 'power2.inOut',
  },
};

// Reducer
function animationReducer(state: AnimationState, action: AnimationAction): AnimationState {
  switch (action.type) {
    case 'PLAY':
      return { ...state, isPlaying: true, isPaused: false };
    
    case 'PAUSE':
      return { ...state, isPlaying: false, isPaused: true };
    
    case 'STOP':
      return { ...state, isPlaying: false, isPaused: false, currentTime: 0, progress: 0 };
    
    case 'SEEK':
      const newTime = Math.max(0, Math.min(action.payload, state.duration));
      return {
        ...state,
        currentTime: newTime,
        progress: state.duration > 0 ? (newTime / state.duration) * 100 : 0,
      };
    
    case 'SET_DURATION':
      return { ...state, duration: action.payload };
    
    case 'SET_PLAYBACK_SPEED':
      return { ...state, playbackSpeed: action.payload };
    
    case 'TOGGLE_LOOP':
      return { ...state, isLooping: !state.isLooping };
    
    case 'TOGGLE_REVERSE':
      return { ...state, isReversed: !state.isReversed };
    
    case 'ADD_KEYFRAME':
      return {
        ...state,
        keyframes: [
          ...state.keyframes,
          { ...action.payload, id: `kf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` },
        ].sort((a, b) => a.time - b.time),
      };
    
    case 'REMOVE_KEYFRAME':
      return {
        ...state,
        keyframes: state.keyframes.filter(kf => kf.id !== action.payload),
      };
    
    case 'UPDATE_KEYFRAME':
      return {
        ...state,
        keyframes: state.keyframes.map(kf =>
          kf.id === action.payload.id ? { ...kf, ...action.payload.updates } : kf
        ).sort((a, b) => a.time - b.time),
      };
    
    case 'ADD_ANIMATION':
      return {
        ...state,
        activeAnimations: [
          ...state.activeAnimations,
          { ...action.payload, id: `anim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` },
        ],
      };
    
    case 'REMOVE_ANIMATION':
      return {
        ...state,
        activeAnimations: state.activeAnimations.filter(anim => anim.id !== action.payload),
      };
    
    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: {
          ...state.settings,
          ...action.payload,
        },
      };
    
    case 'RESET':
      return initialState;
    
    default:
      return state;
  }
}

// Context
const AnimationContext = createContext<{
  state: AnimationState;
  dispatch: React.Dispatch<AnimationAction>;
  actions: {
    play: () => void;
    pause: () => void;
    stop: () => void;
    seek: (time: number) => void;
    setDuration: (duration: number) => void;
    setPlaybackSpeed: (speed: number) => void;
    toggleLoop: () => void;
    toggleReverse: () => void;
    addKeyframe: (keyframe: Omit<AnimationState['keyframes'][0], 'id'>) => void;
    removeKeyframe: (id: string) => void;
    updateKeyframe: (id: string, updates: Partial<AnimationState['keyframes'][0]>) => void;
    addAnimation: (animation: Omit<AnimationState['activeAnimations'][0], 'id'>) => void;
    removeAnimation: (id: string) => void;
    updateSettings: (settings: Partial<AnimationState['settings']>) => void;
    reset: () => void;
  };
  timelineRef: React.MutableRefObject<gsap.core.Timeline | null>;
} | null>(null);

// Provider
export function AnimationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(animationReducer, initialState);
  const timelineRef = useRef<gsap.core.Timeline | null>(null);
  
  // Initialize GSAP timeline
  React.useEffect(() => {
    if (!timelineRef.current) {
      timelineRef.current = gsap.timeline({
        paused: true,
        repeat: state.isLooping ? -1 : 0,
        yoyo: state.isReversed,
        onUpdate: () => {
          if (timelineRef.current) {
            dispatch({
              type: 'SEEK',
              payload: timelineRef.current.time(),
            });
          }
        },
      });
    }
    
    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill();
      }
    };
  }, []);
  
  // Update timeline properties when state changes
  React.useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.repeat(state.isLooping ? -1 : 0);
      timelineRef.current.yoyo(state.isReversed);
      timelineRef.current.timeScale(state.playbackSpeed);
    }
  }, [state.isLooping, state.isReversed, state.playbackSpeed]);
  
  // Actions
  const actions = {
    play: useCallback(() => {
      dispatch({ type: 'PLAY' });
      if (timelineRef.current) {
        timelineRef.current.play();
      }
    }, []),
    
    pause: useCallback(() => {
      dispatch({ type: 'PAUSE' });
      if (timelineRef.current) {
        timelineRef.current.pause();
      }
    }, []),
    
    stop: useCallback(() => {
      dispatch({ type: 'STOP' });
      if (timelineRef.current) {
        timelineRef.current.pause();
        timelineRef.current.seek(0);
      }
    }, []),
    
    seek: useCallback((time: number) => {
      dispatch({ type: 'SEEK', payload: time });
      if (timelineRef.current) {
        timelineRef.current.seek(time);
      }
    }, []),
    
    setDuration: useCallback((duration: number) => {
      dispatch({ type: 'SET_DURATION', payload: duration });
    }, []),
    
    setPlaybackSpeed: useCallback((speed: number) => {
      dispatch({ type: 'SET_PLAYBACK_SPEED', payload: speed });
    }, []),
    
    toggleLoop: useCallback(() => {
      dispatch({ type: 'TOGGLE_LOOP' });
    }, []),
    
    toggleReverse: useCallback(() => {
      dispatch({ type: 'TOGGLE_REVERSE' });
    }, []),
    
    addKeyframe: useCallback((keyframe: Omit<AnimationState['keyframes'][0], 'id'>) => {
      dispatch({ type: 'ADD_KEYFRAME', payload: keyframe });
    }, []),
    
    removeKeyframe: useCallback((id: string) => {
      dispatch({ type: 'REMOVE_KEYFRAME', payload: id });
    }, []),
    
    updateKeyframe: useCallback((id: string, updates: Partial<AnimationState['keyframes'][0]>) => {
      dispatch({ type: 'UPDATE_KEYFRAME', payload: { id, updates } });
    }, []),
    
    addAnimation: useCallback((animation: Omit<AnimationState['activeAnimations'][0], 'id'>) => {
      dispatch({ type: 'ADD_ANIMATION', payload: animation });
    }, []),
    
    removeAnimation: useCallback((id: string) => {
      dispatch({ type: 'REMOVE_ANIMATION', payload: id });
    }, []),
    
    updateSettings: useCallback((settings: Partial<AnimationState['settings']>) => {
      dispatch({ type: 'UPDATE_SETTINGS', payload: settings });
    }, []),
    
    reset: useCallback(() => {
      dispatch({ type: 'RESET' });
      if (timelineRef.current) {
        timelineRef.current.clear();
      }
    }, []),
  };
  
  return (
    <AnimationContext.Provider value={{ state, dispatch, actions, timelineRef }}>
      {children}
    </AnimationContext.Provider>
  );
}

// Hook
export function useAnimation() {
  const context = useContext(AnimationContext);
  if (!context) {
    throw new Error('useAnimation must be used within an AnimationProvider');
  }
  return context;
}