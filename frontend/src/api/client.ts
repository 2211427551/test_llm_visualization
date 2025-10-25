import axios, { AxiosError } from 'axios';
import { ModelForwardResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const cache = new Map<string, ModelForwardResponse>();

export const runModelForward = async (text: string): Promise<ModelForwardResponse> => {
  if (cache.has(text)) {
    return cache.get(text)!;
  }

  try {
    const response = await apiClient.post<ModelForwardResponse>('/model/forward', {
      text,
    });

    cache.set(text, response.data);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{ detail?: string; message?: string }>;
      if (axiosError.response) {
        throw new Error(
          axiosError.response.data?.detail || 
          axiosError.response.data?.message || 
          `Server error: ${axiosError.response.status}`
        );
      } else if (axiosError.request) {
        throw new Error('Backend is not responding. Please ensure the server is running.');
      }
    }
    throw new Error('An unexpected error occurred');
  }
};

export const clearCache = () => {
  cache.clear();
};
