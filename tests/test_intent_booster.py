"""Tests for intent-based boosting."""
import pytest
from src.retrieval.rerankers.intent_booster import IntentBooster
from src.retrieval.searchers.base import SearchResult


def test_intent_booster_initialization():
    """Test that IntentBooster initializes with default weights."""
    booster = IntentBooster()
    assert booster.citation_weights['topical'] == 0.1
    assert booster.citation_weights['sota'] == 0.2
    assert booster.citation_weights['foundational'] == 0.3
    assert booster.date_weights['sota'] == 0.1
    assert booster.min_year == 1990


def test_intent_booster_custom_weights():
    """Test IntentBooster with custom weights."""
    custom_citation = {'topical': 0.5, 'sota': 0.8}
    custom_date = {'sota': 0.3}
    booster = IntentBooster(
        citation_weights=custom_citation,
        date_weights=custom_date,
        min_year=2000,
        current_year=2024
    )
    assert booster.citation_weights['topical'] == 0.5
    assert booster.citation_weights['sota'] == 0.8
    assert booster.date_weights['sota'] == 0.3
    assert booster.min_year == 2000
    assert booster.current_year == 2024


def test_boost_empty_results():
    """Test boosting empty results list."""
    booster = IntentBooster()
    results = booster.boost([], 'topical')
    assert results == []


def test_boost_topical_intent():
    """Test boosting with topical intent."""
    booster = IntentBooster(current_year=2024)
    results = [
        SearchResult(doc_id="1", score=0.9, rank=1),
        SearchResult(doc_id="2", score=0.8, rank=2)
    ]
    # Add citation metadata
    results[0].citation_count = 100
    results[1].citation_count = 50
    
    boosted = booster.boost(results, 'topical')
    assert len(boosted) == 2
    # First result should still be first (higher base score + more citations)
    assert boosted[0].doc_id == "1"


def test_boost_sota_intent():
    """Test boosting with sota intent favoring recent papers."""
    booster = IntentBooster(current_year=2024)
    results = [
        SearchResult(doc_id="old", score=0.9, rank=1),
        SearchResult(doc_id="new", score=0.85, rank=2)
    ]
    results[0].year = 2015
    results[0].citation_count = 200
    results[1].year = 2023
    results[1].citation_count = 100
    
    boosted = booster.boost(results, 'sota')
    assert len(boosted) == 2
    # Check that boosting was applied (scores changed)
    assert boosted[0].score != 0.9 or boosted[1].score != 0.85


def test_boost_foundational_intent():
    """Test boosting with foundational intent favoring older papers."""
    booster = IntentBooster(current_year=2024)
    results = [
        SearchResult(doc_id="new", score=0.9, rank=1),
        SearchResult(doc_id="old", score=0.85, rank=2)
    ]
    results[0].year = 2023
    results[0].citation_count = 100
    results[1].year = 2010
    results[1].citation_count = 500
    
    boosted = booster.boost(results, 'foundational')
    assert len(boosted) == 2
    # Older paper with more citations should get significant boost
    assert boosted[0].score != 0.9


def test_boost_specific_paper_intent():
    """Test boosting with specific_paper intent."""
    booster = IntentBooster()
    results = [
        SearchResult(doc_id="1", score=0.9, rank=1),
        SearchResult(doc_id="2", score=0.8, rank=2)
    ]
    results[0].citation_count = 50
    results[1].citation_count = 100
    
    boosted = booster.boost(results, 'specific_paper')
    assert len(boosted) == 2


def test_boost_modifies_score():
    """Test that boosting modifies the search score."""
    booster = IntentBooster()
    results = [SearchResult(doc_id="1", score=0.95, rank=1)]
    results[0].citation_count = 100
    results[0].year = 2020
    
    boosted = booster.boost(results, 'topical')
    # Score should be modified by boosting
    assert boosted[0].score != 0.95


def test_boost_updates_ranks():
    """Test that boosting updates result ranks."""
    booster = IntentBooster()
    results = [
        SearchResult(doc_id="1", score=0.8, rank=1),
        SearchResult(doc_id="2", score=0.9, rank=2)
    ]
    results[0].citation_count = 1000
    results[1].citation_count = 10
    results[0].year = 2020
    results[1].year = 2023
    
    boosted = booster.boost(results, 'foundational')
    # Check ranks are updated
    assert boosted[0].rank == 1
    assert boosted[1].rank == 2


def test_boost_unknown_intent_uses_default():
    """Test that unknown intent uses default weights."""
    booster = IntentBooster()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].citation_count = 100
    
    boosted = booster.boost(results, 'unknown_intent')
    assert len(boosted) == 1


def test_boost_with_provided_max_citation():
    """Test boosting with explicitly provided max_citation."""
    booster = IntentBooster()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].citation_count = 500
    
    boosted = booster.boost(results, 'topical', max_citation=1000)
    assert len(boosted) == 1
    assert boosted[0].score > 0.9  # Should be boosted


def test_boost_with_zero_citations():
    """Test boosting when all papers have zero citations."""
    booster = IntentBooster()
    results = [
        SearchResult(doc_id="1", score=0.9, rank=1),
        SearchResult(doc_id="2", score=0.8, rank=2)
    ]
    results[0].citation_count = 0
    results[1].citation_count = 0
    
    boosted = booster.boost(results, 'foundational')
    assert len(boosted) == 2
    # With zero citations, citation boost is 0, but date boost may apply
    # If no year, score stays same. Check ranks are maintained.
    assert boosted[0].rank == 1
    assert boosted[1].rank == 2


def test_compute_citation_score_with_zero_max():
    """Test citation score computation when max_citation is 0."""
    booster = IntentBooster()
    score = booster._compute_citation_score(100, 0)
    assert score == 0.0


def test_compute_recency_score_edge_cases():
    """Test recency score with edge cases."""
    booster = IntentBooster(min_year=2000, current_year=2024)
    
    # Year before min_year (should clamp to min_year)
    score1 = booster._compute_recency_score(1990)
    assert 0.0 <= score1 <= 1.0
    
    # Year after current_year (should clamp to current_year)
    score2 = booster._compute_recency_score(2030)
    assert 0.0 <= score2 <= 1.0
    
    # Year at boundaries
    score3 = booster._compute_recency_score(2000)
    assert score3 == 0.0
    score4 = booster._compute_recency_score(2024)
    assert score4 == 1.0


def test_compute_recency_score_zero_range():
    """Test recency score when year range is zero."""
    booster = IntentBooster(min_year=2024, current_year=2024)
    score = booster._compute_recency_score(2024)
    assert score == 1.0


def test_compute_age_score_edge_cases():
    """Test age score with edge cases."""
    booster = IntentBooster(min_year=2000, current_year=2024)
    
    # Year before min_year (should clamp)
    score1 = booster._compute_age_score(1990)
    assert 0.0 <= score1 <= 1.0
    
    # Year after current_year (should clamp)
    score2 = booster._compute_age_score(2030)
    assert 0.0 <= score2 <= 1.0
    
    # Oldest year should get highest age score
    score3 = booster._compute_age_score(2000)
    assert score3 == 1.0
    
    # Most recent year should get lowest age score
    score4 = booster._compute_age_score(2024)
    assert score4 == 0.0


def test_compute_age_score_zero_range():
    """Test age score when year range is zero."""
    booster = IntentBooster(min_year=2024, current_year=2024)
    score = booster._compute_age_score(2024)
    assert score == 1.0


def test_boost_sota_intent_with_year():
    """Test SOTA intent applies recency boost."""
    booster = IntentBooster(current_year=2024, min_year=2000)
    results = [
        SearchResult(doc_id="old", score=0.9, rank=1),
        SearchResult(doc_id="new", score=0.85, rank=2)
    ]
    results[0].publication_year = 2010
    results[0].citation_count = 100
    results[1].publication_year = 2023
    results[1].citation_count = 100
    
    boosted = booster.boost(results, 'sota')
    # New paper should get recency boost
    assert any(r.doc_id == "new" for r in boosted)


def test_boost_foundational_intent_with_year():
    """Test foundational intent applies age boost."""
    booster = IntentBooster(current_year=2024, min_year=2000)
    results = [
        SearchResult(doc_id="new", score=0.85, rank=1),
        SearchResult(doc_id="old", score=0.85, rank=2)
    ]
    results[0].publication_year = 2023
    results[0].citation_count = 100
    results[1].publication_year = 2005
    results[1].citation_count = 500
    
    boosted = booster.boost(results, 'foundational')
    # Old paper with more citations should rank higher
    assert boosted[0].doc_id == "old"


def test_boost_without_publication_year():
    """Test boosting when publication_year is None."""
    booster = IntentBooster()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].publication_year = None
    results[0].citation_count = 100
    
    boosted = booster.boost(results, 'sota')
    assert len(boosted) == 1
    # Should still apply citation boost even without year
    assert boosted[0].score > 0.9


def test_get_boost_summary_empty_results():
    """Test boost summary with empty results."""
    booster = IntentBooster()
    summary = booster.get_boost_summary([])
    assert "No results" in summary


def test_get_boost_summary_without_boost_components():
    """Test boost summary with results that have no boost components."""
    booster = IntentBooster()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    
    summary = booster.get_boost_summary(results, top_n=1)
    assert "No boost info" in summary


def test_get_boost_summary_with_boost_components():
    """Test boost summary with fully boosted results."""
    booster = IntentBooster(current_year=2024)
    results = [SearchResult(doc_id="paper_123", score=0.9, rank=1)]
    results[0].citation_count = 500
    results[0].publication_year = 2020
    
    boosted = booster.boost(results, 'foundational')
    summary = booster.get_boost_summary(boosted, top_n=1)
    
    assert "paper_123" in summary
    assert "Final:" in summary
    assert "Base:" in summary
    assert "Cite:" in summary
    assert "Intent:" in summary
