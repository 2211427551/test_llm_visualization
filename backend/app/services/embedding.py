import numpy as np
from typing import Tuple


class EmbeddingLayer:
    def __init__(self, n_vocab: int, n_embd: int, max_seq_len: int, seed: int = 42):
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        
        np.random.seed(seed)
        self.token_embeddings = np.random.randn(n_vocab, n_embd) * 0.02
        self.positional_encodings = self._create_positional_encoding(max_seq_len, n_embd)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def embed(self, token_ids: list) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = len(token_ids)
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}")
        
        embeddings = self.token_embeddings[token_ids]
        pos_encodings = self.positional_encodings[:seq_len]
        
        return embeddings, pos_encodings
    
    def get_initial_representation(self, token_ids: list) -> np.ndarray:
        embeddings, pos_encodings = self.embed(token_ids)
        return embeddings + pos_encodings
