"""
A4 LLM Judge Integration
Integrates the GRPO-trained judge from Assignment A4 with ArXplorer
"""

import os
import sys
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

# Import A4 components (when available)
try:
    # Add A4 code path
    a4_path = os.path.join(os.path.dirname(__file__), '../../a4/a4_code')
    if os.path.exists(a4_path) and a4_path not in sys.path:
        sys.path.insert(0, a4_path)
    
    from scripts.rl.prompts import build_prompt, parse_answer
    from scripts.rl.reward_fn import compute_reward
    A4_AVAILABLE = True
except ImportError:
    A4_AVAILABLE = False


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation"""
    relevance_score: float      # 0.0 to 1.0
    binary_decision: str        # "yes" or "no"
    confidence: float          # Confidence in judgment
    explanation: str           # Reasoning
    response_time_ms: float    # Time taken
    model_used: str           # Which model made judgment


class OllamaJudge:
    """LLM judge using Ollama (from A4 baseline)"""
    
    def __init__(self, 
                 model: str = "llama3:8b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    async def judge_paper(self, query: str, paper_abstract: str) -> JudgeResult:
        """Judge relevance of paper abstract to query"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Build prompt using A4's prompt format
            if A4_AVAILABLE:
                prompt = build_prompt(query, paper_abstract, max_abstract_chars=1200)
            else:
                prompt = self._fallback_prompt(query, paper_abstract)
            
            # Call Ollama API
            response = await self._call_ollama(prompt)
            
            # Parse response
            if A4_AVAILABLE:
                normalized_answer, format_ok = parse_answer(response)
                if normalized_answer in ["yes", "no"]:
                    relevance_score = 1.0 if normalized_answer == "yes" else 0.0
                    binary_decision = normalized_answer
                    confidence = 0.9 if format_ok else 0.7
                else:
                    relevance_score = 0.5
                    binary_decision = "unknown"
                    confidence = 0.3
            else:
                relevance_score, binary_decision, confidence = self._fallback_parse(response)
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return JudgeResult(
                relevance_score=relevance_score,
                binary_decision=binary_decision,
                confidence=confidence,
                explanation=response[:200] if response else "No explanation",
                response_time_ms=response_time,
                model_used=self.model
            )
            
        except Exception as e:
            self.logger.error(f"Judge failed: {e}")
            return JudgeResult(
                relevance_score=0.5,
                binary_decision="error",
                confidence=0.1,
                explanation=f"Error: {str(e)}",
                response_time_ms=0.0,
                model_used=self.model
            )
    
    def _fallback_prompt(self, query: str, abstract: str) -> str:
        """Fallback prompt when A4 components not available"""
        return (
            "You are a relevance judge. Answer only with 'yes' or 'no' in lowercase.\n"
            "Question: Is the following abstract relevant to the query?\n\n"
            f"Query: {query}\n"
            f"Abstract: {abstract[:1200]}\n\n"
            "Answer (yes/no only):"
        )
    
    def _fallback_parse(self, response: str) -> Tuple[float, str, float]:
        """Fallback parsing when A4 components not available"""
        if not response:
            return 0.5, "unknown", 0.1
        
        response_lower = response.lower().strip()
        if "yes" in response_lower and "no" not in response_lower:
            return 1.0, "yes", 0.8
        elif "no" in response_lower and "yes" not in response_lower:
            return 0.0, "no", 0.8
        else:
            return 0.5, "maybe", 0.5
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent judgments
                        "top_p": 0.9,
                        "max_tokens": 10      # Short responses only
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate", 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '').strip()
                    else:
                        self.logger.error(f"Ollama API error: {response.status}")
                        return ""
        
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            return ""


class GRPOJudge:
    """GRPO-trained judge from A4 (when available)"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load GRPO-trained model"""
        if not A4_AVAILABLE:
            self.logger.warning("A4 components not available, GRPO judge disabled")
            return
        
        try:
            # TODO: Load actual GRPO model from A4
            # This would load the fine-tuned Qwen model from runs/grpo_qwen15b
            # For now, placeholder
            self.logger.info("GRPO model loading not implemented yet")
            
        except Exception as e:
            self.logger.error(f"Failed to load GRPO model: {e}")
    
    async def judge_paper(self, query: str, paper_abstract: str) -> JudgeResult:
        """Judge using GRPO-trained model"""
        if self.model is None:
            self.logger.warning("GRPO model not loaded, falling back to Ollama")
            ollama_judge = OllamaJudge()
            return await ollama_judge.judge_paper(query, paper_abstract)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # TODO: Implement actual GRPO model inference
            # This would use the trained model for yes/no classification
            
            # For now, mock implementation
            relevance_score = 0.8  # Mock score
            binary_decision = "yes" if relevance_score > 0.5 else "no"
            confidence = 0.9
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return JudgeResult(
                relevance_score=relevance_score,
                binary_decision=binary_decision,
                confidence=confidence,
                explanation="GRPO model judgment (mock)",
                response_time_ms=response_time,
                model_used="grpo_qwen15b"
            )
            
        except Exception as e:
            self.logger.error(f"GRPO judge failed: {e}")
            return JudgeResult(
                relevance_score=0.5,
                binary_decision="error",
                confidence=0.1,
                explanation=f"GRPO Error: {str(e)}",
                response_time_ms=0.0,
                model_used="grpo_qwen15b"
            )


class EnsembleJudge:
    """Ensemble of multiple judges for better accuracy"""
    
    def __init__(self):
        self.ollama_judge = OllamaJudge()
        self.grpo_judge = GRPOJudge()
        self.logger = logging.getLogger(__name__)
    
    async def judge_paper(self, query: str, paper_abstract: str) -> JudgeResult:
        """Get judgment from multiple models and combine"""
        
        # Get judgments from all available judges
        tasks = [
            self.ollama_judge.judge_paper(query, paper_abstract),
            self.grpo_judge.judge_paper(query, paper_abstract)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed results
        valid_results = [r for r in results if isinstance(r, JudgeResult)]
        
        if not valid_results:
            return JudgeResult(
                relevance_score=0.5,
                binary_decision="error",
                confidence=0.1,
                explanation="All judges failed",
                response_time_ms=0.0,
                model_used="ensemble"
            )
        
        # Combine results
        avg_score = np.mean([r.relevance_score for r in valid_results])
        avg_confidence = np.mean([r.confidence for r in valid_results])
        total_time = sum([r.response_time_ms for r in valid_results])
        
        # Majority vote for binary decision
        yes_votes = sum(1 for r in valid_results if r.binary_decision == "yes")
        no_votes = sum(1 for r in valid_results if r.binary_decision == "no")
        
        if yes_votes > no_votes:
            binary_decision = "yes"
        elif no_votes > yes_votes:
            binary_decision = "no"
        else:
            binary_decision = "tie"
        
        # Create explanation
        explanations = [f"{r.model_used}: {r.binary_decision}" for r in valid_results]
        explanation = "Ensemble: " + ", ".join(explanations)
        
        return JudgeResult(
            relevance_score=avg_score,
            binary_decision=binary_decision,
            confidence=avg_confidence,
            explanation=explanation,
            response_time_ms=total_time,
            model_used="ensemble"
        )


async def test_judges():
    """Test the different judge implementations"""
    
    print("🧪 Testing LLM Judge Integration")
    print("=" * 50)
    
    query = "machine learning for natural language processing"
    abstract = """
    This paper presents a novel approach to machine learning that combines deep neural networks
    with natural language processing techniques. We demonstrate significant improvements in text
    classification and sentiment analysis tasks using our proposed architecture.
    """
    
    # Test Ollama judge
    print("\n🦙 Testing Ollama Judge:")
    ollama_judge = OllamaJudge()
    result = await ollama_judge.judge_paper(query, abstract)
    print(f"   Decision: {result.binary_decision}")
    print(f"   Score: {result.relevance_score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Time: {result.response_time_ms:.1f}ms")
    print(f"   Explanation: {result.explanation}")
    
    # Test GRPO judge
    print("\n🎯 Testing GRPO Judge:")
    grpo_judge = GRPOJudge()
    result = await grpo_judge.judge_paper(query, abstract)
    print(f"   Decision: {result.binary_decision}")
    print(f"   Score: {result.relevance_score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Time: {result.response_time_ms:.1f}ms")
    print(f"   Model: {result.model_used}")
    
    # Test ensemble
    print("\n🎼 Testing Ensemble Judge:")
    ensemble_judge = EnsembleJudge()
    result = await ensemble_judge.judge_paper(query, abstract)
    print(f"   Decision: {result.binary_decision}")
    print(f"   Score: {result.relevance_score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Time: {result.response_time_ms:.1f}ms")
    print(f"   Explanation: {result.explanation}")
    
    print(f"\n✅ Judge integration test completed!")
    print(f"📍 A4 Components Available: {A4_AVAILABLE}")


if __name__ == "__main__":
    asyncio.run(test_judges())