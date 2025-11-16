"""Dense encoder supporting SPECTER and SPECTER2 models."""

import torch
import numpy as np
from typing import List, Union

from .base import BaseEncoder


class DenseEncoder(BaseEncoder):
    """Dense encoder supporting SPECTER (sentence-transformers) and SPECTER2 (adapters)."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/allenai-specter",
        device: str = None,
        normalize: bool = True,
        use_specter2: bool = False,
        specter2_adapter: str = "allenai/specter2"
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize: Whether to L2-normalize embeddings
            use_specter2: If True, use SPECTER2 with adapters library
            specter2_adapter: Adapter name for SPECTER2 (e.g., 'allenai/specter2' for proximity/retrieval)
        """
        self._model_name = model_name
        self.normalize = normalize
        self.use_specter2 = use_specter2
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        if use_specter2:
            print(f"Loading SPECTER2 with adapter: {specter2_adapter} on {device}")
            self._load_specter2(specter2_adapter)
        else:
            print(f"Loading dense encoder: {model_name} on {device}")
            self._load_sentence_transformer(model_name)
    
    def _load_sentence_transformer(self, model_name: str):
        """Load SPECTER v1 via sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self.tokenizer = None
    
    def _load_specter2(self, adapter_name: str):
        """Load SPECTER2 via adapters library (same as test script)."""
        try:
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "SPECTER2 requires 'adapters' library. Install with:\\n"
                "  pip uninstall peft -y\\n"
                "  pip install transformers==4.38.2 adapters"
            )
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        
        # Load and activate the proximity adapter for retrieval
        adapter_name_loaded = self.model.load_adapter(adapter_name, source="hf", set_active=True)
        print(f"✓ Loaded adapter: {adapter_name_loaded}")
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Explicitly set active adapters after moving to device
        self.model.set_active_adapters(adapter_name_loaded)
        
        # Verify adapter is active
        active = self.model.active_adapters
        print(f"✓ Active adapters: {active}")
        print("Note: Any 'adapters available but none activated' warning is a false positive.")
        
        # SPECTER2 base is BERT-based with 768 dimensions
        self._dimension = 768
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """Encode texts into dense vectors."""
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            if self.use_specter2:
                return self._encode_specter2(texts, batch_size)
            else:
                return self._encode_sentence_transformer(texts, batch_size)
    
    def _encode_sentence_transformer(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using sentence-transformers (SPECTER v1)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)
    
    def _encode_specter2(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using SPECTER2 with adapters (same as test script)."""
        # Suppress false positive warning about adapters
        import warnings
        warnings.filterwarnings('ignore', message='.*adapters available but none are activated.*')
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            output = self.model(**inputs)
            
            # Take [CLS] token embedding (first token)
            embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize if requested
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
