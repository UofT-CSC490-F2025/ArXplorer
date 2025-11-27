"""Qwen reranker for refining search results."""

import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseReranker
from ..searchers.base import SearchResult


class QwenReranker(BaseReranker):
    """Reranker using Qwen3-Reranker models (e.g., Qwen3-Reranker-0.6B)."""
    
    def __init__(
        self,
        doc_texts: Dict[str, str],
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = None,
        max_length: int = 8192,
        batch_size: int = 32,
        instruction: str = None
    ):
        """
        Args:
            doc_texts: Mapping of doc_id → document text
            model_name: HuggingFace Qwen reranker model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length for (query, doc) pairs
            batch_size: Batch size for reranker inference
            instruction: Task instruction (default: web search query retrieval)
        """
        self.doc_texts = doc_texts
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Default instruction for academic paper retrieval
        if instruction is None:
            instruction = (
                "Determine if this academic paper is highly relevant to the query. "
                "Key criteria: (1) For 'original/seminal/first/foundational' queries, papers from before 2018 are more likely original works; "
                "(2) exact title matches are highly relevant; "
                "(3) author name matches matter; "
                "(4) technical queries need papers that directly address core concepts; "
                "(5) deep discussion > passing mention."
            )
        self.instruction = instruction
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading Qwen reranker: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use FP16 on GPU
            low_cpu_mem_usage=True
        )
        self.model.to(device)
        self.model.eval()
        
        # Report model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {param_count / 1e9:.2f}B parameters")
        if device == "cuda":
            print(f"  Using FP16 precision for faster inference")
        
        # Get token IDs for "yes" and "no" (used for scoring)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Precompute prefix and suffix tokens
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    
    def _format_instruction(self, query: str, doc: str) -> str:
        """Format query-doc pair with instruction template."""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Process query-doc pairs into tokenized inputs with prefix/suffix.
        
        Uses fast tokenizer __call__ method directly for better performance.
        """
        # Tokenize without prefix/suffix first (no padding)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            add_special_tokens=False
        )
        
        # Add prefix and suffix tokens to each sequence
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
        
        # Now pad and convert to tensors using prepare_for_model (avoids warning)
        max_len = max(len(ids) for ids in inputs["input_ids"])
        
        padded_inputs = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for ids in inputs["input_ids"]:
            padding_length = max_len - len(ids)
            # Pad on the left (padding_side='left')
            padded_ids = [self.tokenizer.pad_token_id] * padding_length + ids
            attention_mask = [0] * padding_length + [1] * len(ids)
            
            padded_inputs["input_ids"].append(padded_ids)
            padded_inputs["attention_mask"].append(attention_mask)
        
        # Convert to tensors and move to device
        return {
            key: torch.tensor(val, device=self.device) 
            for key, val in padded_inputs.items()
        }
    
    @torch.no_grad()
    def _compute_scores(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute reranking scores from model logits."""
        # Get logits for last token (the model predicts "yes" or "no")
        # use_cache=False to save memory during inference
        batch_scores = self.model(**inputs, use_cache=False).logits[:, -1, :]
        
        # Extract logits for "yes" and "no" tokens
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        # Stack and apply log softmax
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        # Get probability of "yes" (exp of log prob)
        scores = batch_scores[:, 1].exp().tolist()
        
        return scores
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank results using Qwen reranker.
        
        Preserves dense_score and sparse_score from original results,
        and populates cross_encoder_score with the reranker score.
        
        Args:
            query: Query text
            results: Initial search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked SearchResult objects with cross_encoder_score populated
        """
        if not results:
            return []
        
        # Prepare query-document pairs with instruction formatting
        pairs = []
        valid_results = []
        
        for result in results:
            doc_text = self.doc_texts.get(result.doc_id)
            if doc_text:
                formatted_pair = self._format_instruction(query, doc_text)
                pairs.append(formatted_pair)
                valid_results.append(result)
            else:
                # Keep results without text at the end with low scores
                print(f"Warning: No text found for doc_id {result.doc_id}, skipping reranking")
        
        if not pairs:
            print("Warning: No valid document texts found for reranking")
            return results[:top_k]
        
        # Score all pairs with Qwen reranker in batches
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Process inputs (tokenize with prefix/suffix)
            inputs = self._process_inputs(batch_pairs)
            
            # Compute scores
            batch_scores = self._compute_scores(inputs)
            scores.extend(batch_scores)
        
        # Update results with reranker scores (preserve ALL component scores)
        reranked_results = []
        for idx, (result, score) in enumerate(zip(valid_results, scores)):
            new_result = SearchResult(
                doc_id=result.doc_id,
                score=float(score),  # Temporarily set to reranker score
                rank=0,  # Will be set after sorting
                dense_score=result.dense_score,  # Preserve
                sparse_score=result.sparse_score,  # Preserve
                cross_encoder_score=float(score),  # Populate with reranker score
                citation_score=result.citation_score,  # Preserve citation metadata
                citation_count=result.citation_count,  # Preserve citation count
                publication_year=result.publication_year,  # Preserve publication year
                year_score=result.year_score  # Preserve year score
            )
            # Copy metadata attributes
            if hasattr(result, 'title'):
                new_result.title = result.title
            if hasattr(result, 'abstract'):
                new_result.abstract = result.abstract
            if hasattr(result, 'authors'):
                new_result.authors = result.authors
            if hasattr(result, 'categories'):
                new_result.categories = result.categories
            
            reranked_results.append(new_result)
        
        # Sort by reranker score (descending)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked_results[:top_k], 1):
            result.rank = rank
        
        return reranked_results[:top_k]
