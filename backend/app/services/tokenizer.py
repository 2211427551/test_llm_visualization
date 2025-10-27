from typing import List, Tuple, Dict
import re


class SimpleTokenizer:
    def __init__(self, n_vocab: int = 50257):
        self.n_vocab = n_vocab
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._token_counter = 0
    
    def tokenize(self, text: str) -> Tuple[List[int], List[str]]:
        tokens = self._split_text(text)
        token_ids = []
        token_texts = []
        
        for token in tokens:
            if token not in self.vocab:
                if self._token_counter >= self.n_vocab:
                    token_id = self.vocab.get("<UNK>", 0)
                else:
                    token_id = self._token_counter
                    self.vocab[token] = token_id
                    self.inverse_vocab[token_id] = token
                    self._token_counter += 1
            else:
                token_id = self.vocab[token]
            
            token_ids.append(token_id)
            token_texts.append(token)
        
        return token_ids, token_texts
    
    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens if tokens else []
    
    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.inverse_vocab.get(token_id, "<UNK>") for token_id in token_ids]
