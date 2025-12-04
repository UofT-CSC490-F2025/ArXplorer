"""Sparse encoder using SPLADE model."""

import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List, Union, Tuple

from .base import BaseEncoder


class SparseEncoder(BaseEncoder):
    """Sparse encoder using SPLADE."""
    
    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        device: str = None,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length
        """
        self._model_name = model_name
        self.max_length = max_length
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        print(f"Loading sparse encoder: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self._vocab_size = self.model.config.vocab_size
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Encode texts into sparse vectors.
        
        Returns:
            List of (indices, values) tuples for each text
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._encode_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _encode_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Encode a batch of texts."""
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(**tokens)
            
            # SPLADE: log(1 + ReLU(logits)) * attention_mask, then max-pool
            # Compute in steps to avoid keeping large intermediate tensors
            logits = output.logits
            relu_logits = torch.relu(logits)
            log_relu = torch.log(1 + relu_logits)
            
            # Clear intermediate tensors
            del logits, relu_logits, output
            
            # Apply attention mask and max-pool
            attention_mask_expanded = tokens.attention_mask.unsqueeze(-1)
            masked = log_relu * attention_mask_expanded
            splade_vecs = masked.max(dim=1)[0]
            
            # Move to CPU immediately to free GPU memory
            splade_vecs_cpu = splade_vecs.cpu()
            
            # Clear all GPU tensors
            del log_relu, masked, attention_mask_expanded, splade_vecs, tokens
            
            # Force CUDA cache cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Convert to sparse representation (indices + values)
        results = []
        for vec in splade_vecs_cpu:
            indices = vec.nonzero().squeeze().numpy()
            
            # Handle single element case
            if indices.ndim == 0:
                indices = np.array([indices.item()])
            
            values = vec[indices].numpy()
            results.append((indices, values))
        
        return results
    
    def get_dimension(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
