import axios from 'axios';
import { ModelConfig, InitResponse, StepResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const initComputation = async (
  text: string,
  config: ModelConfig
): Promise<InitResponse> => {
  const response = await apiClient.post<InitResponse>('/api/init', {
    text,
    config,
  });
  return response.data;
};

export const fetchStep = async (
  sessionId: string,
  step: number
): Promise<StepResponse> => {
  const response = await apiClient.post<StepResponse>('/api/step', {
    session_id: sessionId,
    step,
  });
  return response.data;
};

export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/api/session/${sessionId}`);
};
