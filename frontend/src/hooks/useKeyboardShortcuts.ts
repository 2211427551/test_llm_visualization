import { useEffect } from 'react';
import { useExecutionStore } from '../store/executionStore';

export const useKeyboardShortcuts = () => {
  const { nextStep, previousStep, isPlaying, setIsPlaying, data } = useExecutionStore();

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (event.key) {
        case 'ArrowRight':
          event.preventDefault();
          nextStep();
          break;
        case 'ArrowLeft':
          event.preventDefault();
          previousStep();
          break;
        case ' ':
          event.preventDefault();
          if (data) {
            setIsPlaying(!isPlaying);
          }
          break;
        case 'Home':
          event.preventDefault();
          useExecutionStore.getState().setCurrentStep(0);
          break;
        case 'End':
          event.preventDefault();
          if (data) {
            useExecutionStore.getState().setCurrentStep(data.steps.length - 1);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [nextStep, previousStep, isPlaying, setIsPlaying, data]);
};
