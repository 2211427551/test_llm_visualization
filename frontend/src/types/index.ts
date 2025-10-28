export interface SparseConfig {
  pattern: 'dense' | 'sliding_window' | 'global_local' | 'blocked' | 'random' | 'custom';
  window_size?: number;
  block_size?: number;
  global_tokens?: number[];
  random_ratio?: number;
  custom_mask?: number[][];
}

export interface ModelConfig {
  n_vocab: number;
  n_embd: number;
  n_layer: number;
  n_head: number;
  d_k: number;
  max_seq_len: number;
  attention_type?: 'standard' | 'sparse';
  sparse_config?: SparseConfig;
}

export interface InitialState {
  embeddings: number[][];
  positional_encodings: number[][];
}

export interface InitResponse {
  session_id: string;
  tokens: number[];
  token_texts: string[];
  total_steps: number;
  initial_state: InitialState;
}

export interface StepResponse {
  step: number;
  step_type: string;
  layer_index: number;
  description: string;
  input_data: number[][];
  output_data: number[][];
  metadata: Record<string, unknown>;
  attention_mask?: number[][];
  sparsity?: number;
}
