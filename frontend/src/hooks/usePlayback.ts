import { useEffect } from 'react';
import { useExecutionStore } from '../store/executionStore';

export const usePlayback = () => {
  const { isPlaying, playbackSpeed, nextStep } = useExecutionStore();

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      nextStep();
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, nextStep]);
};
