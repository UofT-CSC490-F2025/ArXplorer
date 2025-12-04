"""Title and author fuzzy matching for boosting specific paper queries."""

from typing import List, Optional
import re

from ..searchers.base import SearchResult


class TitleAuthorMatcher:
    """
    Fast fuzzy matching component for boosting papers with similar titles or authors.
    
    Uses token-based matching instead of Levenshtein for speed:
    - Title matching: Jaccard similarity of normalized tokens
    - Author matching: Substring/token matching on last names
    
    Much faster than character-level edit distance while still effective for
    "specific_paper" and "foundational" intent queries.
    """
    
    def __init__(
        self,
        title_threshold: float = 0.5,  # Lower threshold since we use tokens now
        author_threshold: float = 0.7,
        title_boost_weight: float = 1.0,
        author_boost_weight: float = 1.0
    ):
        """
        Args:
            title_threshold: Minimum Jaccard similarity for title match (0-1)
            author_threshold: Minimum token overlap for author match (0-1)
            title_boost_weight: Boost to add for title match
            author_boost_weight: Boost to add for author match
        """
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold
        self.title_boost_weight = title_boost_weight
        self.author_boost_weight = author_boost_weight
    
    def match_and_boost(
        self,
        results: List[SearchResult],
        target_title: Optional[str] = None,
        target_authors: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Apply fuzzy matching boost to results based on title and author similarity.
        
        Args:
            results: List of search results to boost
            target_title: Target paper title from LLM extraction (or None)
            target_authors: Target author names from LLM extraction (or None)
            
        Returns:
            Results with boosted scores and updated ranks
        """
        if not target_title and not target_authors:
            # Nothing to match against
            return results
        
        # Normalize target strings once (not per result)
        target_title_norm = None
        target_authors_norm = None
        
        if target_title:
            target_title_norm = self._normalize_string(target_title)
        
        if target_authors:
            target_authors_norm = [self._normalize_string(a) for a in target_authors]
        
        # Apply matching to each result (modify in-place for speed)
        match_count = 0
        for result in results:
            title_match = False
            author_match = False
            title_score = 0.0
            author_score = 0.0
            
            # Check title match using token-based Jaccard similarity (fast)
            if target_title_norm and hasattr(result, 'title') and result.title:
                candidate_title_norm = self._normalize_string(result.title)
                similarity = self._jaccard_similarity(target_title_norm, candidate_title_norm)
                
                if similarity >= self.title_threshold:
                    title_match = True
                    title_score = similarity
            
            # Check author match using fast substring/token matching
            if target_authors_norm and hasattr(result, 'authors') and result.authors:
                # Parse authors from result (handle various formats)
                candidate_authors = self._parse_authors(result.authors)
                candidate_authors_norm = [self._normalize_string(a) for a in candidate_authors]
                
                # Check if any target author matches any candidate author (fast substring/token match)
                best_author_similarity = 0.0
                for target_author in target_authors_norm:
                    # Extract tokens from target
                    target_tokens = set(target_author.split())
                    
                    for candidate_author in candidate_authors_norm:
                        candidate_tokens = set(candidate_author.split())
                        
                        # Check for substring match (fast) or token overlap
                        if target_author in candidate_author or candidate_author in target_author:
                            similarity = 1.0
                        else:
                            # Jaccard similarity of tokens
                            similarity = self._jaccard_similarity_tokens(target_tokens, candidate_tokens)
                        
                        # Check if this is a match
                        if similarity >= self.author_threshold:
                            author_match = True
                            author_score = similarity
                            break
                        
                        # Track best similarity even if not a match
                        if similarity > best_author_similarity:
                            best_author_similarity = similarity
                    
                    # Break outer loop if we found a match
                    if author_match:
                        break
                
                # If no match but we have a best score, use it
                if not author_match and best_author_similarity > 0:
                    author_score = best_author_similarity
            
            # Apply boost in-place (much faster than creating new objects)
            if title_match or author_match:
                boost = 0.0
                if title_match:
                    boost += self.title_boost_weight
                if author_match:
                    boost += self.author_boost_weight
                
                result.score += boost
                
                # Store matching info for debugging
                if not hasattr(result, 'boost_components'):
                    result.boost_components = {}
                
                result.boost_components['title_match'] = title_match
                result.boost_components['title_score'] = title_score
                result.boost_components['author_match'] = author_match
                result.boost_components['author_score'] = author_score
                result.boost_components['match_boost'] = boost
                match_count += 1
        
        # Sort by score and update ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        return results
    
    def _normalize_string(self, s: str) -> str:
        """Normalize string for matching: lowercase, strip whitespace, remove punctuation."""
        s = s.lower().strip()
        # Remove common punctuation but keep spaces
        s = re.sub(r'[^a-z0-9\s]', '', s)
        # Collapse multiple spaces
        s = re.sub(r'\s+', ' ', s)
        return s
    
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Fast Jaccard similarity between two strings (token-based)."""
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        return self._jaccard_similarity_tokens(tokens1, tokens2)
    
    def _jaccard_similarity_tokens(self, tokens1: set, tokens2: set) -> float:
        """Jaccard similarity between two sets of tokens."""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    def _parse_authors(self, authors) -> List[str]:
        """
        Parse authors from various formats.
        
        Args:
            authors: Can be string, list of strings, or list of dicts
            
        Returns:
            List of author name strings
        """
        if isinstance(authors, str):
            # Split by common delimiters
            if ',' in authors:
                return [a.strip() for a in authors.split(',')]
            elif ';' in authors:
                return [a.strip() for a in authors.split(';')]
            elif ' and ' in authors:
                return [a.strip() for a in authors.split(' and ')]
            else:
                return [authors.strip()]
        
        elif isinstance(authors, list):
            parsed = []
            for author in authors:
                if isinstance(author, str):
                    parsed.append(author.strip())
                elif isinstance(author, dict):
                    # Handle dict format like {'name': 'John Smith'}
                    name = author.get('name') or author.get('full_name') or author.get('author')
                    if name:
                        parsed.append(str(name).strip())
            return parsed
        
        return []
    
    def get_match_summary(self, result: SearchResult) -> Optional[str]:
        """Get human-readable summary of title/author matching for a result."""
        if not hasattr(result, 'boost_components'):
            return None
        
        bc = result.boost_components
        if not bc.get('title_match') and not bc.get('author_match'):
            return None
        
        parts = []
        if bc.get('title_match'):
            parts.append(f"Title match: {bc.get('title_score', 0):.2f}")
        if bc.get('author_match'):
            parts.append(f"Author match: {bc.get('author_score', 0):.2f}")
        
        if parts:
            return f"Match boost: +{bc.get('match_boost', 0):.2f} ({', '.join(parts)})"
        
        return None
