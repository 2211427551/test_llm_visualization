/**
 * Playback control store
 * Manages animation playback state and controls
 */

import { create } from 'zustand';

export interface Step {
  id: string;
  name: string;
  description: string;
  duration: number;
  formula?: string;
}

interface PlaybackState {
  // State
  isPlaying: boolean;
  currentStep: number;
  totalSteps: number;
  speed: number;
  progress: number; // 0-1 progress within current step
  steps: Step[];
  
  // Actions
  play: () => void;
  pause: () => void;
  toggle: () => void;
  nextStep: () => void;
  prevStep: () => void;
  goToStep: (step: number) => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  setProgress: (progress: number) => void;
  setSteps: (steps: Step[]) => void;
  updateProgress: (delta: number) => void;
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  // Initial state
  isPlaying: false,
  currentStep: 0,
  totalSteps: 0,
  speed: 1,
  progress: 0,
  steps: [],
  
  // Actions
  play: () => set({ isPlaying: true }),
  
  pause: () => set({ isPlaying: false }),
  
  toggle: () => set((state) => ({ isPlaying: !state.isPlaying })),
  
  nextStep: () => set((state) => {
    const nextStep = Math.min(state.currentStep + 1, state.totalSteps - 1);
    return {
      currentStep: nextStep,
      progress: 0,
      isPlaying: nextStep < state.totalSteps - 1 ? state.isPlaying : false,
    };
  }),
  
  prevStep: () => set((state) => ({
    currentStep: Math.max(state.currentStep - 1, 0),
    progress: 0,
  })),
  
  goToStep: (step: number) => set((state) => ({
    currentStep: Math.max(0, Math.min(step, state.totalSteps - 1)),
    progress: 0,
  })),
  
  reset: () => set({
    currentStep: 0,
    progress: 0,
    isPlaying: false,
  }),
  
  setSpeed: (speed: number) => set({ speed }),
  
  setProgress: (progress: number) => set({ progress: Math.max(0, Math.min(1, progress)) }),
  
  setSteps: (steps: Step[]) => set({
    steps,
    totalSteps: steps.length,
    currentStep: 0,
    progress: 0,
  }),
  
  updateProgress: (delta: number) => set((state) => {
    const newProgress = state.progress + delta * state.speed;
    
    if (newProgress >= 1) {
      // Move to next step
      const nextStep = state.currentStep + 1;
      if (nextStep < state.totalSteps) {
        return {
          currentStep: nextStep,
          progress: 0,
        };
      } else {
        // Reached the end
        return {
          progress: 1,
          isPlaying: false,
        };
      }
    }
    
    return { progress: newProgress };
  }),
}));

/**
 * Hook to automatically update progress while playing
 */
export function usePlaybackLoop(fps: number = 60) {
  const { isPlaying, updateProgress } = usePlaybackStore();
  
  if (typeof window !== 'undefined') {
    const frameTime = 1000 / fps;
    let lastTime = Date.now();
    let animationFrameId: number;
    
    const loop = () => {
      if (isPlaying) {
        const currentTime = Date.now();
        const delta = (currentTime - lastTime) / 1000; // Convert to seconds
        lastTime = currentTime;
        
        updateProgress(delta);
        animationFrameId = requestAnimationFrame(loop);
      }
    };
    
    if (isPlaying) {
      lastTime = Date.now();
      animationFrameId = requestAnimationFrame(loop);
    }
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }
}
