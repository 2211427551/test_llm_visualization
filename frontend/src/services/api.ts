import { z } from 'zod';

// API Request/Response Schemas
export const InitRequestSchema = z.object({
  text: z.string().min(1, "Text cannot be empty"),
  config: z.object({
    n_vocab: z.number().min(1),
    n_embd: z.number().min(1),
    n_layer: z.number().min(1),
    n_head: z.number().min(1),
    max_seq_len: z.number().min(1),
  }),
});

export const StepRequestSchema = z.object({
  session_id: z.string(),
  step: z.number().min(0),
});

export type InitRequest = z.infer<typeof InitRequestSchema>;
export type StepRequest = z.infer<typeof StepRequestSchema>;

export interface InitResponse {
  session_id: string;
  tokens: number[];
  token_texts: string[];
  total_steps: number;
  initial_state: {
    embeddings: number[][];
    positional_encodings: number[][];
  };
}

export interface StepResponse {
  step: number;
  step_type: string;
  layer_index: number;
  description: string;
  input_data: number[][];
  output_data: number[][] | number[];
  metadata: Record<string, unknown>;
}

// API Client Class
export class TransformerAPI {
  private baseURL: string;
  private controller: AbortController | null = null;

  constructor(baseURL: string = '/api') {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Cancel any previous request
    if (this.controller) {
      this.controller.abort();
    }
    
    this.controller = new AbortController();
    
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      ...options,
      signal: this.controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request was cancelled');
      }
      throw error;
    } finally {
      this.controller = null;
    }
  }

  // Initialize a new session
  async initSession(request: InitRequest): Promise<InitResponse> {
    // Validate request
    const validatedRequest = InitRequestSchema.parse(request);
    
    return this.request<InitResponse>('/init', {
      method: 'POST',
      body: JSON.stringify(validatedRequest),
    });
  }

  // Get a specific step
  async getStep(request: StepRequest): Promise<StepResponse> {
    // Validate request
    const validatedRequest = StepRequestSchema.parse(request);
    
    return this.request<StepResponse>('/step', {
      method: 'POST',
      body: JSON.stringify(validatedRequest),
    });
  }

  // Stream steps (Server-Sent Events)
  async *streamSteps(sessionId: string): AsyncGenerator<StepResponse, void, unknown> {
    const url = `${this.baseURL}/stream/${sessionId}`;
    
    try {
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data;
            } catch (error) {
              console.warn('Failed to parse SSE data:', line, error);
            }
          }
        }
      }
    } catch (error) {
      throw error;
    }
  }

  // Delete a session
  async deleteSession(sessionId: string): Promise<{ message: string; session_id: string }> {
    return this.request(`/session/${sessionId}`, {
      method: 'DELETE',
    });
  }

  // List active sessions
  async listSessions(): Promise<{
    total_sessions: number;
    session_ids: string[];
  }> {
    return this.request('/sessions');
  }

  // Get health check
  async getHealth(): Promise<{
    status: string;
    active_sessions: number;
    cached_computations: number;
    cache_hit_rate: string;
  }> {
    return this.request('/health');
  }

  // Get cache statistics
  async getCacheStats(): Promise<{
    cache_size: number;
    max_cache_size: number;
    cached_keys: string[];
  }> {
    return this.request('/cache/stats');
  }

  // Clear cache
  async clearCache(): Promise<{
    message: string;
    remaining_entries: number;
  }> {
    return this.request('/cache/clear', {
      method: 'DELETE',
    });
  }

  // Cancel any ongoing request
  cancelRequest(): void {
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
  }
}

// Create default API instance
export const api = new TransformerAPI();

// Utility functions for common operations
export async function initializeTransformer(
  text: string,
  config: InitRequest['config'] = {
    n_vocab: 50257,
    n_embd: 768,
    n_layer: 6,
    n_head: 8,
    max_seq_len: 512,
  }
): Promise<InitResponse> {
  return api.initSession({ text, config });
}

export async function getTransformerStep(
  sessionId: string,
  step: number
): Promise<StepResponse> {
  return api.getStep({ session_id: sessionId, step });
}

export async function* streamTransformerSteps(
  sessionId: string
): AsyncGenerator<StepResponse, void, unknown> {
  yield* api.streamSteps(sessionId);
}