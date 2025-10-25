"""
Simple tokenizers for text processing.
Supports character-level and BPE-like word-level tokenization.
"""
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str
    id: int
    start_pos: int
    end_pos: int


class CharTokenizer:
    """
    Character-level tokenizer that converts text to individual characters.
    Useful for fine-grained analysis and small vocabulary size.
    """
    
    def __init__(self, max_vocab_size: int = 256):
        """
        Initialize character tokenizer.
        
        Args:
            max_vocab_size: Maximum vocabulary size (default: 256 for ASCII)
        """
        self.max_vocab_size = max_vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary (ASCII characters)."""
        for i in range(min(256, self.max_vocab_size)):
            char = chr(i)
            self.char_to_id[char] = i
            self.id_to_char[i] = char
    
    def encode(self, text: str) -> Tuple[List[int], List[TokenInfo]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (token_ids, token_info_list)
        """
        token_ids = []
        token_info = []
        
        for pos, char in enumerate(text):
            char_id = self.char_to_id.get(char, 0)  # Use 0 for unknown chars
            token_ids.append(char_id)
            token_info.append(TokenInfo(
                text=char,
                id=char_id,
                start_pos=pos,
                end_pos=pos + 1
            ))
        
        return token_ids, token_info
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        chars = [self.id_to_char.get(tid, '?') for tid in token_ids]
        return ''.join(chars)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char_to_id)


class SimpleBPETokenizer:
    """
    Simple BPE-like word-level tokenizer.
    Tokenizes by spaces and punctuation, similar to basic BPE.
    """
    
    def __init__(self, max_vocab_size: int = 10000):
        """
        Initialize BPE tokenizer.
        
        Args:
            max_vocab_size: Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.next_id = 4
    
    def _split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into words/tokens with position tracking.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, start_pos, end_pos) tuples
        """
        import re
        # Split on whitespace and punctuation while keeping punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = []
        for match in re.finditer(pattern, text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> Tuple[List[int], List[TokenInfo]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Tuple of (token_ids, token_info_list)
        """
        words = self._split_text(text)
        token_ids = []
        token_info = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_id["<BOS>"])
            token_info.append(TokenInfo(
                text="<BOS>",
                id=self.word_to_id["<BOS>"],
                start_pos=0,
                end_pos=0
            ))
        
        for word, start_pos, end_pos in words:
            # Add word to vocabulary if not seen before
            if word not in self.word_to_id:
                if self.next_id < self.max_vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
            
            word_id = self.word_to_id.get(word, self.word_to_id["<UNK>"])
            token_ids.append(word_id)
            token_info.append(TokenInfo(
                text=word,
                id=word_id,
                start_pos=start_pos,
                end_pos=end_pos
            ))
        
        if add_special_tokens:
            text_len = len(text)
            token_ids.append(self.word_to_id["<EOS>"])
            token_info.append(TokenInfo(
                text="<EOS>",
                id=self.word_to_id["<EOS>"],
                start_pos=text_len,
                end_pos=text_len
            ))
        
        return token_ids, token_info
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        words = []
        for tid in token_ids:
            word = self.id_to_word.get(tid, "<UNK>")
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)
        return ' '.join(words)
    
    @property
    def vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.word_to_id)


def get_tokenizer(tokenizer_type: str = "bpe", **kwargs):
    """
    Factory function to get a tokenizer instance.
    
    Args:
        tokenizer_type: Type of tokenizer ("char" or "bpe")
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "char":
        return CharTokenizer(**kwargs)
    elif tokenizer_type == "bpe":
        return SimpleBPETokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
