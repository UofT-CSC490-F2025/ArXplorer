"""LLM-based query rewriter using instruction-tuned models."""

import torch
import json
import re
from typing import Optional, List, Dict, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from .base import BaseQueryRewriter


def build_milvus_filter_expr(filters: Dict, enable_citation_filters: bool = True, enable_year_filters: bool = True) -> Optional[str]:
    """
    Convert filter dict to Milvus boolean expression string.
    
    Args:
        filters: Dict with structure:
            {
                'year': {'min': int|None, 'max': int|None},
                'citation_count': {'min': int|None, 'max': None}
            }
        enable_citation_filters: Whether to include citation_count filters
        enable_year_filters: Whether to include year filters
    
    Returns:
        Milvus filter expression string or None if no filters
        
    Example:
        filters = {'year': {'min': 2020, 'max': None}, 'citation_count': {'min': 100, 'max': None}}
        returns: "year >= 2020 and citation_count >= 100"
    """
    if not filters:
        return None
    
    conditions = []
    
    # Year filters (if enabled)
    if enable_year_filters:
        year_filter = filters.get('year', {})
        if year_filter:
            year_min = year_filter.get('min')
            year_max = year_filter.get('max')
            
            if year_min is not None:
                conditions.append(f"year >= {year_min}")
            if year_max is not None:
                conditions.append(f"year <= {year_max}")
    
    # Citation count filters (if enabled)
    if enable_citation_filters:
        citation_filter = filters.get('citation_count', {})
        if citation_filter:
            citation_min = citation_filter.get('min')
            citation_max = citation_filter.get('max')
            
            if citation_min is not None:
                conditions.append(f"citation_count >= {citation_min}")
            if citation_max is not None:
                conditions.append(f"citation_count <= {citation_max}")
    
    if not conditions:
        return None
    
    return " and ".join(conditions)


# ---------------------------------------------------------------------------
# Part 3 coverage notes (see tests/test_query_rewriting.py):
#   1. test_llm_rewriter_parses_canonical_json – happy-path canonical JSON.
#   2. test_llm_rewriter_falls_back_to_pattern_detection – JSON parse failure.
#   3. test_llm_rewriter_defaults_to_cpu – device auto-detection branch.
#   4. test_build_prompt_without_chat_template / test_build_prompt_canonical_no_chat_template –
#      prompt fallback when tokenizer lacks chat templates.
#   5. test_rewrite_uses_default_num_rewrites – num_rewrites clamp/padding.
#   6. test_rewrite_empty_generated_text_defaults_to_query – empty decode fallback.
#   7. test_rewrite_causal_lm_without_canonical – legacy non-canonical mode.
#   8. test_extract_filters_and_rewrite_success / failure / invalid_structure /
#      seq2seq_mode – filter parsing success/fallback branches.
#   9. test_extract_filters_no_chat_template – filter prompt fallback branch.
#  10. test_detect_canonical_fallback_false – pattern matcher negative case.
# These edge-case tests ensure both positive and negative flows are exercised.
# ---------------------------------------------------------------------------
class LLMQueryRewriter(BaseQueryRewriter):
    """Query rewriter using local LLM for query expansion."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 0.3,
        num_rewrites: int = 1,
        canonical_intent_enabled: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum length of rewritten query
            temperature: Sampling temperature (lower = more deterministic)
            num_rewrites: Default number of rewrites to generate
            canonical_intent_enabled: Whether to detect canonical intent
        """
        self._model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.num_rewrites = num_rewrites
        self.canonical_intent_enabled = canonical_intent_enabled
        
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
        
        # Canonical intent patterns (fallback if LLM fails)
        self.canonical_patterns = [
            r'\b(original|seminal|first|founding|pioneering)\b.*\bpaper\b',
            r'\b(resnet|unet|bert|gpt|transformer|gan|vae|lstm|attention)\b.*\bpaper\b',
            r'\bpaper\b.*\b(introduced|proposed|invented)\b',
            r'\bwho\s+(invented|created|proposed|introduced)\b'
        ]
    
    def _detect_canonical_fallback(self, query: str) -> bool:
        """
        Fallback pattern-based canonical intent detection.
        
        Args:
            query: User query string
            
        Returns:
            True if query matches canonical patterns
        """
        query_lower = query.lower()
        for pattern in self.canonical_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def _build_prompt(self, query: str, num_rewrites: int = 1) -> str:
        """Build prompt for the LLM."""
        if self.is_causal_lm:
            # Use chat template for instruction-tuned models
            if self.canonical_intent_enabled:
                system_msg = (
                    f"You are an expert academic search assistant. Your goal is to optimize a user's search query for an arXiv paper retrieval pipeline."
                    f"You must analyze the user's query and output a JSON object with two fields: is_canonical and rewritten_queries."
                    f"1. is_canonical (boolean):"
                    f"- Set to TRUE if the user is looking for a specific, seminal, or original paper (e.g., original unet paper, attention is all you need, paper introducing resnet)."
                    f"- Set to FALSE if the user is asking a general question, looking for a survey, or comparing methods (e.g., how does unet work, unet vs resnet, graph neural net applications)."
                    f"2. rewritten_queries (list of strings):"
                    f"- Generate EXACTLY {num_rewrites} search queries to improve retrieval recall."
                    f"- CRITICAL: If is_canonical is TRUE, you must attempt to include the specific title of the paper or the authors if you know them."
                    f"- If is_canonical is FALSE, focus on expanding keywords and synonyms."
                    f"Output ONLY the raw JSON object. Do not output markdown formatting or explanation."
                )
                user_msg = (
                    f"The user query is:\n\n{query}\n\n"
                    f"Output ONLY the raw JSON object. Do not output markdown formatting or explanation."
                )
            else:
                # Legacy mode without canonical intent
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
    
    def rewrite(self, query: str, num_rewrites: Optional[int] = None) -> Union[List[str], Dict[str, Union[bool, List[str]]]]:
        """
        Rewrite query using LLM with optional canonical intent detection.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewrites to generate (uses instance default if None)
            
        Returns:
            If canonical_intent_enabled:
                Dict with keys: 'canonical_intent' (bool), 'rewrites' (List[str])
            Else:
                List[str] of rewritten queries (backward compatibility)
        """
        if num_rewrites is None:
            num_rewrites = self.num_rewrites
        
        # Clamp to reasonable range
        num_rewrites = max(1, min(num_rewrites, 10))
        
        # Build prompt
        prompt = self._build_prompt(query, num_rewrites)
        
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
                    max_new_tokens=self.max_length if not self.canonical_intent_enabled else 256,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9,
                    num_return_sequences=1,  # Generate once for JSON output
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
        
        if self.canonical_intent_enabled and self.is_causal_lm:
            # Parse JSON output for canonical intent
            output = outputs[0]
            prompt_length = inputs['input_ids'].shape[1]
            generated_ids = output[prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Try to parse JSON
            canonical_intent = False
            rewrites = []
            
            try:
                # Extract JSON from output (may have extra text)
                json_match = re.search(r'\{[^{}]*"is_canonical"[^{}]*"rewritten_queries"[^{}]*\}', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    canonical_intent = bool(result.get('is_canonical', False))
                    rewrites = result.get('rewritten_queries', [])
                    
                    # Ensure we have the right number of rewrites
                    if len(rewrites) < num_rewrites:
                        # Pad with original query if needed
                        rewrites.extend([query] * (num_rewrites - len(rewrites)))
                    rewrites = rewrites[:num_rewrites]
                else:
                    raise ValueError("No JSON found in output")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback: use pattern-based detection and treat output as single rewrite
                print(f"Warning: Failed to parse JSON from LLM ({e}). Using fallback.")
                canonical_intent = self._detect_canonical_fallback(query)
                # Use generated text as rewrite if it looks reasonable
                if generated_text and len(generated_text.strip()) > 0 and len(generated_text) < 200:
                    rewrites = [generated_text.strip()]
                else:
                    rewrites = [query]
                # Generate additional rewrites if needed
                rewrites.extend([query] * (num_rewrites - len(rewrites)))
                rewrites = rewrites[:num_rewrites]
            
            # Apply fallback canonical detection if LLM said False but patterns match
            if not canonical_intent:
                canonical_intent = self._detect_canonical_fallback(query)
            
            return {
                'canonical_intent': canonical_intent,
                'rewrites': rewrites
            }
        else:
            # Legacy mode or seq2seq: return list of rewrites
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
    
    def extract_filters_and_rewrite(self, query: str, num_rewrites: Optional[int] = None, current_year: int = 2025) -> Dict:
        """
        Extract search filters and rewrite query in a single LLM call.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewrites to generate
            current_year: Current year for dynamic date ranges
            
        Returns:
            Dict with keys:
                - filters: Dict with year (min/max) and citation_count (min/max)
                - confidence: Float 0-1 indicating filter confidence
                - rewrites: List[str] of rewritten queries
        """
        if num_rewrites is None:
            num_rewrites = self.num_rewrites
        
        num_rewrites = max(1, min(num_rewrites, 10))
        
        # Build combined prompt for filter extraction + rewriting
        if self.is_causal_lm:
            system_msg = (
                f"You are an expert academic search assistant. Analyze the user's query and extract search filters + generate rewritten queries.\n\n"
                f"Current year: {current_year}\n\n"
                f"Extract:\n"
                f"1. year constraints (min/max or null if not relevant)\n"
                f"2. citation_count minimum (choose from: 100, 500, 1000, 5000, or null)\n"
                f"3. confidence (0-1) in the extracted filters\n"
                f"4. {num_rewrites} rewritten search queries\n\n"
                f"Guidelines:\n"
                f"- 'original/seminal/foundational' → early year + high citations (≥500 or ≥1000)\n"
                f"- 'recent/latest/new' → year ≥ {current_year - 5}\n"
                f"- 'state-of-the-art/SOTA' → year ≥ {current_year - 2}, citations ≥ 50\n"
                f"- Specific paper names → try to recall publication year\n"
                f"- General topics → no filters (all null)\n\n"
                f"Output ONLY raw JSON (no markdown formatting):\n"
                f"{{\n"
                f'  "filters": {{"year": {{"min": <int|null>, "max": <int|null>}}, "citation_count": {{"min": <int|null>, "max": null}}}},\n'
                f'  "confidence": <float>,\n'
                f'  "rewrites": [<{num_rewrites} strings>]\n'
                f"}}"
            )
            
            user_msg = f'Query: "{query}"\n\nOutput JSON:'
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            # Use chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = f"{system_msg}\n\nUser: {user_msg}\nAssistant:"
        else:
            # Fallback for seq2seq models - just do query rewriting
            return {
                'filters': {
                    'year': {'min': None, 'max': None},
                    'citation_count': {'min': None, 'max': None}
                },
                'confidence': 0.0,
                'rewrites': self.rewrite(query, num_rewrites)
            }
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Parse output
        output = outputs[0]
        prompt_length = inputs['input_ids'].shape[1]
        generated_ids = output[prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Try to parse JSON
        try:
            # Extract JSON from output (handle markdown code blocks)
            generated_text = generated_text.strip()
            if generated_text.startswith('```'):
                # Remove markdown code blocks
                generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
                generated_text = re.sub(r'\s*```$', '', generated_text)
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Extract and validate filters
                filters = result.get('filters', {})
                year_filter = filters.get('year', {'min': None, 'max': None})
                citation_filter = filters.get('citation_count', {'min': None, 'max': None})
                
                # Ensure proper structure
                if not isinstance(year_filter, dict):
                    year_filter = {'min': None, 'max': None}
                if not isinstance(citation_filter, dict):
                    citation_filter = {'min': None, 'max': None}
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                rewrites = result.get('rewrites', [query])
                if not isinstance(rewrites, list) or len(rewrites) == 0:
                    rewrites = [query]
                
                return {
                    'filters': {
                        'year': year_filter,
                        'citation_count': citation_filter
                    },
                    'confidence': confidence,
                    'rewrites': rewrites[:num_rewrites]
                }
            else:
                raise ValueError("No JSON object found")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to parse filter JSON ({e}). Returning no filters.")
            # Fallback: no filters, just return query
            return {
                'filters': {
                    'year': {'min': None, 'max': None},
                    'citation_count': {'min': None, 'max': None}
                },
                'confidence': 0.0,
                'rewrites': [query]
            }
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
