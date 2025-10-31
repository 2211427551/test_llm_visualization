'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useVisualization } from '@/contexts/VisualizationContext';
import { useAnimation } from '@/contexts/AnimationContext';
import { initializeTransformer, getTransformerStep, streamTransformerSteps } from '@/services/api';
import { InitRequest } from '@/services/api';

interface UseTransformerAPIOptions {
  text: string;
  config?: InitRequest['config'];
  autoPlay?: boolean;
  onStepComplete?: (step: number, totalSteps: number) => void;
  onError?: (error: Error) => void;
}

export function useTransformerAPI({
  text,
  config,
  autoPlay = false,
  onStepComplete,
  onError,
}: UseTransformerAPIOptions) {
  const { state: vizState, actions: vizActions } = useVisualization();
  const { actions: animActions } = useAnimation();
  
  const sessionIdRef = useRef<string | null>(null);
  const isInitializedRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize transformer session
  const initialize = useCallback(async () => {
    if (!text.trim()) {
      throw new Error('Text cannot be empty');
    }

    try {
      // Cancel any ongoing initialization
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      abortControllerRef.current = new AbortController();

      // Initialize session with backend
      const response = await initializeTransformer(text, config);
      
      sessionIdRef.current = response.session_id;
      isInitializedRef.current = true;

      // Update visualization state with initial data
      vizActions.loadData({
        totalSteps: response.total_steps,
        currentStep: 0,
        modules: {
          ...vizState.modules,
          embedding: {
            ...vizState.modules.embedding,
            isActive: true,
            tokens: response.token_texts,
            embeddings: response.initial_state.embeddings,
            positionalEncodings: response.initial_state.positional_encodings,
          },
        },
      });

      // Set up animation timeline
      animActions.setDuration(response.total_steps);
      
      // Add keyframes for each step
      response.token_texts.forEach((token, index) => {
        animActions.addKeyframe({
          time: index,
          label: `Token ${index + 1}: ${token}`,
          description: `Processing token "${token}"`,
        });
      });

      // Auto-play if requested
      if (autoPlay) {
        animActions.play();
      }

      return response;
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError?.(error);
      }
      throw error;
    }
  }, [text, config, autoPlay, vizActions, animActions, vizState.modules, onError]);

  // Get a specific step
  const getStep = useCallback(async (step: number) => {
    if (!sessionIdRef.current || !isInitializedRef.current) {
      throw new Error('Session not initialized');
    }

    try {
      const stepData = await getTransformerStep(sessionIdRef.current, step);
      
      // Update visualization state with step data
      vizActions.setStep(step);
      
      // Update module states based on step type
      const stepType = stepData.step_type;
      
      if (stepType.includes('attention')) {
        vizActions.updateModule('attention', {
          isActive: true,
          attentionWeights: Array.isArray(stepData.output_data[0]) ? stepData.output_data as number[][] : [stepData.output_data as number[]],
        });
      }
      
      if (stepType.includes('ffn') || stepType.includes('moe')) {
        vizActions.updateModule('moe', {
          isActive: true,
        });
      }
      
      if (stepType.includes('output')) {
        vizActions.updateModule('output', {
          isActive: true,
          logits: Array.isArray(stepData.output_data[0]) ? stepData.output_data[0] : [],
        });
      }

      onStepComplete?.(step, vizState.totalSteps);
      
      return stepData;
    } catch (error) {
      onError?.(error as Error);
      throw error;
    }
  }, [sessionIdRef, isInitializedRef, vizActions, vizState.totalSteps, onStepComplete, onError]);

  // Stream all steps
  const streamSteps = useCallback(async () => {
    if (!sessionIdRef.current || !isInitializedRef.current) {
      throw new Error('Session not initialized');
    }

    try {
      for await (const stepData of streamTransformerSteps(sessionIdRef.current)) {
        const step = stepData.step;
        
        // Update visualization state
        vizActions.setStep(step);
        
        // Update module states
        const stepType = stepData.step_type;
        
        if (stepType.includes('attention')) {
          vizActions.updateModule('attention', {
            isActive: true,
            attentionWeights: Array.isArray(stepData.output_data[0]) ? stepData.output_data as number[][] : [stepData.output_data as number[]],
          });
        }
        
        if (stepType.includes('ffn') || stepType.includes('moe')) {
          vizActions.updateModule('moe', {
            isActive: true,
          });
        }
        
        if (stepType.includes('output')) {
          vizActions.updateModule('output', {
            isActive: true,
            logits: Array.isArray(stepData.output_data[0]) ? stepData.output_data[0] : [],
          });
        }

        onStepComplete?.(step, vizState.totalSteps);
        
        // Update animation timeline
        animActions.seek(step);
      }
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError?.(error);
      }
      throw error;
    }
  }, [sessionIdRef, isInitializedRef, vizActions, animActions, vizState.totalSteps, onStepComplete, onError]);

  // Reset and cleanup
  const reset = useCallback(() => {
    // Cancel any ongoing operations
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Clear session
    sessionIdRef.current = null;
    isInitializedRef.current = false;
    
    // Reset visualization state
    vizActions.reset();
    animActions.reset();
  }, [vizActions, animActions]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    initialize,
    getStep,
    streamSteps,
    reset,
    isInitialized: isInitializedRef.current,
    sessionId: sessionIdRef.current,
  };
}