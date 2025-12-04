"""Base abstract classes for encoders."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for text encoders."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """
        Encode text(s) into vector representation(s).
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (n_texts, embedding_dim) for dense
            or sparse representations
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension/vocabulary size of the encoder."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
