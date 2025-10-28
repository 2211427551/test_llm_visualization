export interface ModelConfig {
  n_vocab: number;
  n_embd: number;
  n_layer: number;
  n_head: number;
  d_k: number;
  max_seq_len: number;
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
}
