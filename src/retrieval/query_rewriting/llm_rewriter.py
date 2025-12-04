"""LLM-based query rewriter using instruction-tuned models."""

import torch
import json
import re
from typing import Optional, List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseQueryRewriter

# Optional OpenAI client for vLLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional boto3 for AWS Bedrock
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


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


class LLMQueryRewriter(BaseQueryRewriter):
    """Query rewriter using LLM for query expansion - supports local, vLLM API, and AWS Bedrock."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 0.3,
        num_rewrites: int = 1,
        canonical_intent_enabled: bool = True,
        use_vllm: bool = False,
        vllm_endpoint: str = "http://localhost:8000/v1",
        vllm_timeout: int = 30,
        use_bedrock: bool = False,
        bedrock_model_id: str = "mistral.mistral-7b-instruct-v0:2",
        bedrock_region: str = "ca-central-1",
        bedrock_max_tokens: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model identifier (local/vLLM only)
            device: Device to use ('cuda', 'cpu', or None for auto) - local only
            max_length: Maximum length of rewritten query
            temperature: Sampling temperature (lower = more deterministic)
            num_rewrites: Default number of rewrites to generate
            canonical_intent_enabled: Whether to detect canonical intent
            use_vllm: Use vLLM API instead of loading model locally
            vllm_endpoint: vLLM server endpoint (OpenAI-compatible API)
            vllm_timeout: Timeout for vLLM API calls in seconds
            use_bedrock: Use AWS Bedrock API instead of local/vLLM
            bedrock_model_id: Bedrock model ID (e.g., mistral.mistral-7b-instruct-v0:2)
            bedrock_region: AWS region for Bedrock
            bedrock_max_tokens: Max tokens for Bedrock response
        """
        self._model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.num_rewrites = num_rewrites
        self.canonical_intent_enabled = canonical_intent_enabled
        self.use_vllm = use_vllm
        self.vllm_endpoint = vllm_endpoint
        self.vllm_timeout = vllm_timeout
        self.use_bedrock = use_bedrock
        self.bedrock_model_id = bedrock_model_id
        self.bedrock_region = bedrock_region
        self.bedrock_max_tokens = bedrock_max_tokens
        
        # Initialize Bedrock, vLLM, or local model
        if use_bedrock:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 library required for AWS Bedrock. Install with: pip install boto3")
            
            print(f"Initializing AWS Bedrock client: {bedrock_model_id} in {bedrock_region}")
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=bedrock_region
            )
            self.model = None
            self.tokenizer = None
            self.client = None
            self.device = "bedrock"
            self.is_causal_lm = True
            print(f"✓ AWS Bedrock client initialized (model: {bedrock_model_id})")
        elif use_vllm:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library required for vLLM. Install with: pip install openai")
            
            print(f"Initializing vLLM client: {vllm_endpoint}")
            self.client = OpenAI(
                base_url=vllm_endpoint,
                api_key="dummy",  # vLLM doesn't require authentication
                timeout=vllm_timeout
            )
            self.model = None
            self.tokenizer = None
            self.device = "api"
            self.is_causal_lm = True  # vLLM serves causal LMs
            print(f"✓ vLLM client initialized (model: {model_name})")
        else:
            # Local model loading
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.client = None
            
            print(f"Loading query rewriting model: {model_name} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load causal LM
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
            print(f"✓ Query rewriter loaded (causal LM)")
    
    def rewrite(self, query: str, num_rewrites: int = 1) -> List[str]:
        """
        Legacy method for backward compatibility with BaseQueryRewriter.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewritten queries to generate
            
        Returns:
            List of rewritten queries
        """
        # Delegate to the new structured method and extract rewrites
        from datetime import datetime
        result = self.extract_intent_filters_and_rewrite(
            query,
            num_rewrites=num_rewrites,
            current_year=datetime.now().year
        )
        return result.get('rewrites', [query])
    
    def extract_intent_filters_and_rewrite(self, query: str, num_rewrites: Optional[int] = None, current_year: int = 2025) -> Dict:
        """
        Extract intent, search filters, and rewritten queries in a single LLM call.
        
        Args:
            query: Original user query
            num_rewrites: Number of rewrites to generate
            current_year: Current year for dynamic date ranges
            
        Returns:
            Dict with keys:
                - intent: str (topical|sota|foundational|comparison|method_lookup|specific_paper)
                - year_constraint: Dict with min/max or None
                - citation_threshold: int or None
                - rewrites: List[str] of rewritten queries (length = num_rewrites)
        """
        if num_rewrites is None:
            num_rewrites = self.num_rewrites
        
        num_rewrites = max(1, min(num_rewrites, 10))
        
        # Build structured prompt
        system_msg = (
            f"You are an expert academic search assistant. Analyze the user's query and extract structured search information.\n\n"
            f"Current year: {current_year}\n\n"
            f"Output a JSON object with these fields:\n"
            f"{{\n"
            f'  "intent": "<topical|sota|foundational|comparison|method_lookup|specific_paper>",\n'
            f'  "year_constraint": {{"min": <int|null>, "max": <int|null>}},\n'
            f'  "citation_threshold": <int|null>,\n'
            f'  "target_title": "<string|null>",\n'
            f'  "target_authors": ["<string>", ...] or null,\n'
            f'  "rewrites": [<{num_rewrites} search query strings>]\n'
            f"}}\n\n"
            f"Intent definitions:\n"
            f"- topical: General topic exploration\n"
            f"- sota: Looking for state-of-the-art/recent methods\n"
            f"- foundational: Looking for original/seminal papers\n"
            f"- comparison: Comparing multiple methods\n"
            f"- method_lookup: Understanding how something works\n"
            f"- specific_paper: Looking for a specific known paper\n\n"
            f"Guidelines:\n"
            f"- For 'original/seminal/first' queries → intent: foundational, citation_threshold: 500-1000\n"
            f"- For 'recent/new/advances/SOTA' queries → intent: sota, year_constraint: {{min: {current_year-3}}}\n"
            f"- For 'vs/versus/comparison' queries → intent: comparison\n"
            f"- For 'what is/how does/explain' queries → intent: method_lookup\n"
            f"- If year mentioned explicitly → set year_constraint\n"
            f"- If query is foundational or specific paper → set year contraints to that paper IF KNOWN\n"
            f"- If specific paper (e.g., 'ResNet paper') → intent: specific_paper\n\n"
            f"Target Title/Authors Extraction:\n"
            f"- ONLY extract target_title if the query is VERY CLEAR about a specific paper\n"
            f"- If you know the exact title, write it\n"
            f"- If only partial title is given, infer the full title ONLY if you are certain\n"
            f"- For target_authors, list author last names (expand from last name if mentioned)\n"
            f"- Set to null if uncertain or query is not about a specific paper\n\n"
            f"Rewrites:\n"
            f"- Generate EXACTLY {num_rewrites} diverse search queries\n"
            f"- CRITICAL: For foundational/specific_paper intents, you MUST include the exact paper title and authors in the rewrite if known.\n"
            f"- For other intents, expand with synonyms and related terms\n"
            f"- Vary the phrasing to maximize retrieval recall\n\n"
            f"Output ONLY valid JSON. No markdown, no explanation."
        )
        
        user_msg = f"Analyze this query: {query}"
        
        # Use Bedrock, vLLM, or local model
        if self.use_bedrock:
            return self._extract_structured_bedrock(system_msg, user_msg, query, num_rewrites)
        elif self.use_vllm:
            return self._extract_structured_vllm(system_msg, user_msg, query, num_rewrites)
        else:
            return self._extract_structured_local(system_msg, user_msg, query, num_rewrites)
    
    def _extract_structured_bedrock(self, system_msg: str, user_msg: str, query: str, num_rewrites: int) -> Dict:
        """Extract structured query using AWS Bedrock API."""
        try:
            # Format prompt for Mistral instruction format
            prompt = f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
            
            # Format request payload
            native_request = {
                "prompt": prompt,
                "max_tokens": self.bedrock_max_tokens,
                "temperature": self.temperature,
            }
            
            # Invoke Bedrock model
            response = self.bedrock_client.invoke_model(
                modelId=self.bedrock_model_id,
                body=json.dumps(native_request)
            )
            
            # Parse response
            model_response = json.loads(response["body"].read())
            generated_text = model_response["outputs"][0]["text"]
            
            return self._parse_structured_response(generated_text, query, num_rewrites)
            
        except (ClientError, Exception) as e:
            print(f"Warning: AWS Bedrock API call failed ({e}). Using fallback.")
            return self._fallback_structured_response(query, num_rewrites)
    
    def _extract_structured_vllm(self, system_msg: str, user_msg: str, query: str, num_rewrites: int) -> Dict:
        """Extract structured query using vLLM API."""
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=self.temperature,
                max_tokens=400,
                top_p=0.9,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            
            generated_text = response.choices[0].message.content
            return self._parse_structured_response(generated_text, query, num_rewrites)
            
        except Exception as e:
            print(f"Warning: vLLM API call failed ({e}). Using fallback.")
            return self._fallback_structured_response(query, num_rewrites)
    
    def _extract_structured_local(self, system_msg: str, user_msg: str, query: str, num_rewrites: int) -> Dict:
        """Extract structured query using local model."""
        if not self.is_causal_lm:
            # Seq2seq fallback
            return self._fallback_structured_response(query, num_rewrites)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Use chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            prompt = f"{system_msg}\n\nUser: {user_msg}\nAssistant:"
        
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
                max_new_tokens=400,
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
        
        return self._parse_structured_response(generated_text, query, num_rewrites)
    
    def _parse_structured_response(self, generated_text: str, query: str, num_rewrites: int) -> Dict:
        """Parse JSON response from LLM for structured query."""
        try:
            # Clean markdown formatting
            generated_text = generated_text.strip()
            if generated_text.startswith('```'):
                generated_text = re.sub(r'^```(?:json)?\s*', '', generated_text)
                generated_text = re.sub(r'\s*```$', '', generated_text)
            
            # Extract JSON - try to parse the text directly first
            json_str = generated_text.strip()
            
            # If text doesn't start with {, try to find JSON object
            if not json_str.startswith('{'):
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', generated_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON found")
                json_str = json_match.group(0)
            
            # Try to parse just the JSON object (stop at first complete object)
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # If that fails, try to find the first complete JSON object
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(json_str)
            
            # Validate and extract fields
            intent = result.get('intent', 'topical')
            valid_intents = {'topical', 'sota', 'foundational', 'comparison', 'method_lookup', 'specific_paper'}
            if intent not in valid_intents:
                intent = 'topical'
            
            year_constraint = result.get('year_constraint', {'min': None, 'max': None})
            if not isinstance(year_constraint, dict):
                year_constraint = {'min': None, 'max': None}
            
            citation_threshold = result.get('citation_threshold')
            if citation_threshold is not None:
                try:
                    citation_threshold = int(citation_threshold)
                    if citation_threshold < 0:
                        citation_threshold = None
                except (ValueError, TypeError):
                    citation_threshold = None
            
            rewrites = result.get('rewrites', [query])
            if not isinstance(rewrites, list) or len(rewrites) == 0:
                rewrites = [query]
            # Pad or truncate to exactly num_rewrites
            while len(rewrites) < num_rewrites:
                rewrites.append(query)
            rewrites = rewrites[:num_rewrites]
            
            # Extract target_title and target_authors
            target_title = result.get('target_title')
            if target_title and not isinstance(target_title, str):
                target_title = None
            if target_title == "" or target_title == "null":
                target_title = None
            
            target_authors = result.get('target_authors')
            if target_authors:
                if isinstance(target_authors, list):
                    # Filter out non-strings and empty strings
                    target_authors = [a for a in target_authors if isinstance(a, str) and a.strip()]
                    if not target_authors:
                        target_authors = None
                else:
                    target_authors = None
            
            return {
                'intent': intent,
                'year_constraint': year_constraint,
                'citation_threshold': citation_threshold,
                'target_title': target_title,
                'target_authors': target_authors,
                'rewrites': rewrites
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to parse structured JSON ({e}). Using fallback.")
            return self._fallback_structured_response(query, num_rewrites)
    
    def _fallback_structured_response(self, query: str, num_rewrites: int) -> Dict:
        """Fallback response when LLM fails."""
        # Simple pattern-based intent detection
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['original', 'seminal', 'first', 'foundational', 'pioneering']):
            intent = 'foundational'
        elif any(kw in query_lower for kw in ['recent', 'latest', 'new', 'advances', 'state-of-the-art', 'sota']):
            intent = 'sota'
        elif any(kw in query_lower for kw in ['vs', 'versus', 'compare', 'comparison', 'difference between']):
            intent = 'comparison'
        elif any(kw in query_lower for kw in ['what is', 'how does', 'explain', 'definition']):
            intent = 'method_lookup'
        elif 'paper' in query_lower:
            intent = 'specific_paper'
        else:
            intent = 'topical'
        
        return {
            'intent': intent,
            'year_constraint': {'min': None, 'max': None},
            'citation_threshold': None,
            'target_title': None,
            'target_authors': None,
            'rewrites': [query] * num_rewrites
        }
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
