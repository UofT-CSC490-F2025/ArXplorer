"""
LLM-based relevance judge using qwen2.5-1.5b-instruct.
Evaluates whether a paper is relevant to a given query.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import re


class LLMJudge:
    """Lightweight LLM for judging paper relevance."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = None):
        """
        Initialize the LLM judge.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu), auto-detected if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading LLM judge: {model_name} on {device}...")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.model.eval()
        
        print(f"✓ LLM judge loaded")
    
    def judge_relevance(self, query: str, paper_title: str, paper_abstract: str = "") -> Dict:
        """
        Judge if a paper is relevant to a query.
        
        Args:
            query: User search query
            paper_title: Paper title
            paper_abstract: Paper abstract (optional, improves accuracy)
            
        Returns:
            Dictionary with 'relevant' (bool), 'confidence' (float), 'reasoning' (str)
        """
        # Construct prompt
        abstract_text = f"\n\nAbstract: {paper_abstract[:500]}" if paper_abstract else ""
        
        prompt = f"""Is this paper relevant to the query?

Query: {query}

Paper: {paper_title}{abstract_text}

Answer ONLY "Yes" or "No":"""
        
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1,  # Yes/No is a single token
                temperature=0.1,
                do_sample=False  # Deterministic for consistency
            )
        
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        # Parse response
        relevant = self._parse_relevance(response)
        
        return {
            'relevant': relevant,
            'confidence': 1.0 if relevant else 0.0,
            'reasoning': response  # Just Yes/No now
        }
    
    def _parse_relevance(self, response: str) -> bool:
        """
        Parse LLM response to extract binary relevance.
        
        Looks for "Yes" or "No" at the start of the response.
        """
        response_lower = response.lower().strip()
        
        # Check first word/phrase
        if response_lower.startswith("yes"):
            return True
        elif response_lower.startswith("no"):
            return False
        
        # Fallback: check if "yes" appears before "no"
        yes_pos = response_lower.find("yes")
        no_pos = response_lower.find("no")
        
        if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
            return True
        
        # Default to not relevant if ambiguous
        return False
    
    def batch_judge(
        self, 
        query: str, 
        papers: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Judge relevance for multiple papers (processes sequentially).
        
        Args:
            query: Search query
            papers: List of dicts with 'title' and optionally 'abstract'
            
        Returns:
            List of relevance judgments
        """
        results = []
        for paper in papers:
            judgment = self.judge_relevance(
                query,
                paper.get('title', ''),
                paper.get('abstract', '')
            )
            results.append(judgment)
        
        return results


def main():
    """Test the LLM judge."""
    print("Testing LLM Judge...")
    print()
    
    judge = LLMJudge()
    
    # Test case
    query = "transformer architecture for natural language processing"
    paper_title = "Attention Is All You Need"
    paper_abstract = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    
    result = judge.judge_relevance(query, paper_title, paper_abstract)
    
    print(f"Query: {query}")
    print(f"Paper: {paper_title}")
    print(f"Relevant: {result['relevant']}")
    print(f"Reasoning: {result['reasoning']}")
    print()
    
    # Negative test case
    query2 = "deep learning for computer vision"
    paper_title2 = "A Survey of Economic Growth Models"
    
    result2 = judge.judge_relevance(query2, paper_title2)
    
    print(f"Query: {query2}")
    print(f"Paper: {paper_title2}")
    print(f"Relevant: {result2['relevant']}")
    print(f"Reasoning: {result2['reasoning']}")


if __name__ == "__main__":
    main()
