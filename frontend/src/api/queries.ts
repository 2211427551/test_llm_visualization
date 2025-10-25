import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runModelForward } from './client';
import type { ModelForwardResponse } from '../types';

export const useModelForwardMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (text: string) => runModelForward(text),
    onSuccess: (data: ModelForwardResponse) => {
      queryClient.setQueryData(['model-forward', data], data);
    },
  });
};
