"""LLM-based query rewriter using instruction-tuned models."""

import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from .base import BaseQueryRewriter


class LLMQueryRewriter(BaseQueryRewriter):
    """Query rewriter using local LLM for query expansion."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 0.3,
        num_rewrites: int = 1
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum length of rewritten query
            temperature: Sampling temperature (lower = more deterministic)
            num_rewrites: Default number of rewrites to generate
        """
        self._model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.num_rewrites = num_rewrites
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading query rewriting model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Detect model type and load accordingly
        if "t5" in model_name.lower() or "flan" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_causal_lm = False
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self.is_causal_lm = True
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Query rewriter loaded (type: {'causal LM' if self.is_causal_lm else 'seq2seq'})")
    
    def _build_prompt(self, query: str) -> str:
        """Build prompt for the LLM."""
        if self.is_causal_lm:
            # Use chat template for instruction-tuned models
            system_msg = (
                "You are an expert at rewriting search queries for academic paper retrieval. "
                "Expand abbreviations, add technical terms, and include relevant paper titles when known. "
                "Keep the rewrite concise (under 20 words)."
            )
            user_msg = f"Rewrite this search query for finding academic papers: {query}"
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Use chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format
                return f"{system_msg}\n\nUser: {user_msg}\nAssistant:"
        else:
            # Seq2seq models (FLAN-T5) - use few-shot examples
            prompt = (
                "Rewrite search queries to find academic papers. Expand abbreviations and add key terms.\n\n"
                "Query: original resnet paper\n"
                "Rewritten: Deep Residual Learning for Image Recognition ResNet convolutional neural network\n\n"
                "Query: bert nlp\n"
                "Rewritten: BERT Pre-training of Deep Bidirectional Transformers for Language Understanding natural language processing\n\n"
                "Query: gan generative adversarial network\n"
                "Rewritten: Generative Adversarial Networks GAN Goodfellow neural network image generation\n\n"
                f"Query: {query}\n"
                "Rewritten:"
            )
            return prompt
    
    def rewrite(self, query: str, num_rewrites: Optional[int] = None) -> List[str]:
        """
        Rewrite query using LLM.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewrites to generate (uses instance default if None)
            
        Returns:
            List of expanded/rewritten queries
        """
        if num_rewrites is None:
            num_rewrites = self.num_rewrites
        
        # Clamp to reasonable range
        num_rewrites = max(1, min(num_rewrites, 10))
        
        # Build prompt
        prompt = self._build_prompt(query)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512 if self.is_causal_lm else 512,
            truncation=True
        ).to(self.device)
        
        # Generate rewritten queries
        with torch.no_grad():
            if self.is_causal_lm:
                # Causal LM: generate continuation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9,
                    num_return_sequences=num_rewrites,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Seq2seq: generate full output
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9,
                    num_return_sequences=num_rewrites,
                    num_beams=max(num_rewrites, 2)  # Use beam search for diversity
                )
        
        # Decode all rewrites
        rewrites = []
        for output in outputs:
            if self.is_causal_lm:
                # Skip the prompt tokens for causal LM
                prompt_length = inputs['input_ids'].shape[1]
                generated_ids = output[prompt_length:]
                rewritten = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                # Decode full output for seq2seq
                rewritten = self.tokenizer.decode(output, skip_special_tokens=True)
            
            rewrites.append(rewritten.strip())
        
        return rewrites
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
