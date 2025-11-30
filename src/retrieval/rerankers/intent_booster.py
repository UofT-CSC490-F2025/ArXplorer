"""Intent-based boosting for post-RRF ranking."""

import math
from typing import List, Dict, Optional
from datetime import datetime

from ..searchers.base import SearchResult


class IntentBooster:
    """
    Apply intent-specific boosting to search results after RRF fusion.
    
    Boosts results based on:
    1. Citation count (normalized log scale)
    2. Publication year (recency for sota, age for foundational)
    
    Intent-specific strategies:
    - topical/comparison/method_lookup/default: Light citation boost (0.1)
    - sota: Medium citation boost (0.2) + recency boost (0.1)
    - foundational: Heavy citation boost (0.3) + age boost (0.1)
    """
    
    def __init__(
        self,
        citation_weights: Optional[Dict[str, float]] = None,
        date_weights: Optional[Dict[str, float]] = None,
        min_year: int = 1990,
        current_year: Optional[int] = None
    ):
        """
        Args:
            citation_weights: Citation boost weight per intent (default provided)
            date_weights: Date boost weight per intent (default provided)
            min_year: Minimum year in corpus (for normalization)
            current_year: Current year (defaults to now)
        """
        # Default citation weights
        self.citation_weights = citation_weights or {
            'topical': 0.1,
            'comparison': 0.1,
            'method_lookup': 0.1,
            'default': 0.1,
            'sota': 0.2,
            'foundational': 0.3,
            'specific_paper': 0.15
        }
        
        # Default date weights
        self.date_weights = date_weights or {
            'topical': 0.0,
            'comparison': 0.0,
            'method_lookup': 0.0,
            'default': 0.0,
            'sota': 0.1,          # Favor recent
            'foundational': 0.1,   # Favor older
            'specific_paper': 0.0
        }
        
        self.min_year = min_year
        self.current_year = current_year or datetime.now().year
        self.year_range = self.current_year - self.min_year
    
    def boost(
        self,
        results: List[SearchResult],
        intent: str,
        max_citation: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Apply intent-based boosting to search results.
        
        Args:
            results: List of SearchResult from initial search
            intent: Detected intent (topical, sota, foundational, etc.)
            max_citation: Maximum citation count for normalization (computed if None)
            
        Returns:
            List of SearchResult with updated scores and rankings
        """
        if not results:
            return results
        
        # Normalize intent
        intent = intent.lower()
        if intent not in self.citation_weights:
            intent = 'default'
        
        # Get boost weights for this intent
        citation_weight = self.citation_weights[intent]
        date_weight = self.date_weights[intent]
        
        # Compute max citation for normalization if not provided
        if max_citation is None:
            max_citation = max((r.citation_count or 0) for r in results)
            if max_citation == 0:
                max_citation = 1  # Avoid division by zero
        
        # Apply boosting
        for result in results:
            base_score = result.score  # RRF score from Milvus
            
            # Citation boost: log-normalized to [0, 1]
            citation_score = self._compute_citation_score(
                result.citation_count or 0,
                max_citation
            )
            citation_boost = citation_weight * citation_score
            
            # Date boost (intent-specific)
            date_boost = 0.0
            if date_weight > 0.0 and result.publication_year is not None:
                if intent == 'sota':
                    # Favor recent papers
                    date_score = self._compute_recency_score(result.publication_year)
                elif intent == 'foundational':
                    # Favor older papers
                    date_score = self._compute_age_score(result.publication_year)
                else:
                    date_score = 0.0
                
                date_boost = date_weight * date_score
            
            # Combined boosted score
            result.score = base_score + citation_boost + date_boost
            
            # Store component scores for analysis
            if not hasattr(result, 'boost_components'):
                result.boost_components = {}
            result.boost_components['base_rrf'] = base_score
            result.boost_components['citation_boost'] = citation_boost
            result.boost_components['date_boost'] = date_boost
            result.boost_components['intent'] = intent
        
        # Re-sort by boosted scores
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        return results
    
    def _compute_citation_score(self, citation_count: int, max_citation: int) -> float:
        """
        Normalize citation count to [0, 1] using log scale.
        
        Formula: log(citation + 1) / log(max_citation + 1)
        
        Args:
            citation_count: Number of citations
            max_citation: Maximum citations in result set
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_citation == 0:
            return 0.0
        
        return math.log(citation_count + 1) / math.log(max_citation + 1)
    
    def _compute_recency_score(self, year: int) -> float:
        """
        Compute recency score favoring recent papers.
        
        Formula: (year - min_year) / year_range
        
        Args:
            year: Publication year
            
        Returns:
            Score in [0, 1] where 1.0 = most recent
        """
        if self.year_range == 0:
            return 1.0
        
        # Clamp to valid range
        year = max(self.min_year, min(self.current_year, year))
        
        return (year - self.min_year) / self.year_range
    
    def _compute_age_score(self, year: int) -> float:
        """
        Compute age score favoring older papers.
        
        Formula: (max_year - year) / year_range
        
        Args:
            year: Publication year
            
        Returns:
            Score in [0, 1] where 1.0 = oldest
        """
        if self.year_range == 0:
            return 1.0
        
        # Clamp to valid range
        year = max(self.min_year, min(self.current_year, year))
        
        return (self.current_year - year) / self.year_range
    
    def get_boost_summary(self, results: List[SearchResult], top_n: int = 5) -> str:
        """
        Get human-readable summary of boosting applied to top results.
        
        Args:
            results: Boosted results
            top_n: Number of top results to summarize
            
        Returns:
            Formatted string summary
        """
        if not results:
            return "No results to summarize"
        
        summary_lines = [f"Boost Summary (Top {top_n}):"]
        summary_lines.append("-" * 70)
        
        for i, result in enumerate(results[:top_n], 1):
            if not hasattr(result, 'boost_components'):
                summary_lines.append(f"{i}. {result.doc_id}: No boost info")
                continue
            
            bc = result.boost_components
            summary_lines.append(
                f"{i}. {result.doc_id[:30]:30} | "
                f"Final: {result.score:.4f} | "
                f"Base: {bc.get('base_rrf', 0):.4f} | "
                f"Cite: +{bc.get('citation_boost', 0):.4f} | "
                f"Date: +{bc.get('date_boost', 0):.4f} | "
                f"Intent: {bc.get('intent', 'unknown')}"
            )
        
        return "\n".join(summary_lines)


# Example usage
if __name__ == "__main__":
    # Mock results
    results = [
        SearchResult(
            doc_id="paper_1",
            score=0.8,  # RRF score
            rank=1,
            citation_count=5000,
            publication_year=2017,
            dense_score=None,
            sparse_score=None
        ),
        SearchResult(
            doc_id="paper_2",
            score=0.75,
            rank=2,
            citation_count=500,
            publication_year=2023,
            dense_score=None,
            sparse_score=None
        ),
        SearchResult(
            doc_id="paper_3",
            score=0.7,
            rank=3,
            citation_count=10000,
            publication_year=2015,
            dense_score=None,
            sparse_score=None
        )
    ]
    
    booster = IntentBooster()
    
    print("Testing intent-based boosting")
    print("=" * 70)
    
    # Test foundational intent (favor older + highly cited)
    print("\nIntent: foundational")
    foundational_results = booster.boost(results.copy(), 'foundational')
    print(booster.get_boost_summary(foundational_results, 3))
    
    # Test SOTA intent (favor recent + cited)
    print("\nIntent: sota")
    sota_results = booster.boost(results.copy(), 'sota')
    print(booster.get_boost_summary(sota_results, 3))
    
    # Test topical intent (light citation boost only)
    print("\nIntent: topical")
    topical_results = booster.boost(results.copy(), 'topical')
    print(booster.get_boost_summary(topical_results, 3))
