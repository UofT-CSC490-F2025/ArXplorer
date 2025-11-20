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
        specter2_base_adapter: str = "allenai/specter2",
        specter2_query_adapter: str = "allenai/specter2_adhoc_query"
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize: Whether to L2-normalize embeddings
            use_specter2: If True, use SPECTER2 with adapters library
            specter2_base_adapter: Adapter for document embeddings (e.g., 'allenai/specter2')
            specter2_query_adapter: Adapter for query embeddings (e.g., 'allenai/specter2_adhoc_query')
        """
        self._model_name = model_name
        self.normalize = normalize
        self.use_specter2 = use_specter2
        self.specter2_base_adapter = specter2_base_adapter
        self.specter2_query_adapter = specter2_query_adapter
        self.current_adapter = None
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        if use_specter2:
            print(f"Loading SPECTER2 with base adapter: {specter2_base_adapter} on {device}")
            self._load_specter2(specter2_base_adapter)
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
        
        # Load base model and tokenizer (only on first call)
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
            self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            self.model.to(self.device)
            self.model.eval()
            self._dimension = 768
            self.loaded_adapters = {}
        
        # Load adapter if not already loaded
        if adapter_name not in self.loaded_adapters:
            adapter_name_loaded = self.model.load_adapter(adapter_name, source="hf", set_active=False)
            self.loaded_adapters[adapter_name] = adapter_name_loaded
            print(f"✓ Loaded adapter: {adapter_name_loaded}")
            # Move adapter parameters to the correct device
            self.model.to(self.device)
        
        # Activate the adapter
        self.model.set_active_adapters(self.loaded_adapters[adapter_name])
        self.current_adapter = adapter_name
        
        # Verify adapter is active
        active = self.model.active_adapters
        print(f"✓ Active adapter: {active}")
        print("Note: Any 'adapters available but none activated' warning is a false positive.")
    
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
    
    def set_adapter(self, adapter_name: str):
        """Switch to a different SPECTER2 adapter (e.g., for query vs document encoding)."""
        if not self.use_specter2:
            print("Warning: set_adapter called but use_specter2=False")
            return
        
        if adapter_name == self.current_adapter:
            return  # Already using this adapter
        
        # Load and activate the adapter
        if adapter_name not in self.loaded_adapters:
            adapter_name_loaded = self.model.load_adapter(adapter_name, source="hf", set_active=False)
            self.loaded_adapters[adapter_name] = adapter_name_loaded
            print(f"✓ Loaded adapter: {adapter_name_loaded}")
            # Ensure adapter parameters are on the correct device
            self.model.to(self.device)
        
        self.model.set_active_adapters(self.loaded_adapters[adapter_name])
        self.current_adapter = adapter_name
        print(f"✓ Switched to adapter: {adapter_name}")
    
    def use_query_adapter(self):
        """Switch to query adapter for encoding queries."""
        self.set_adapter(self.specter2_query_adapter)
    
    def use_base_adapter(self):
        """Switch to base adapter for encoding documents."""
        self.set_adapter(self.specter2_base_adapter)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
