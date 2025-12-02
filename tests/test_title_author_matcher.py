"""Tests for title and author fuzzy matching."""
import pytest
from src.retrieval.rerankers.title_author_matcher import TitleAuthorMatcher
from src.retrieval.searchers.base import SearchResult


def test_title_author_matcher_initialization():
    """Test that TitleAuthorMatcher initializes with default parameters."""
    matcher = TitleAuthorMatcher()
    assert matcher.title_threshold == 0.5
    assert matcher.author_threshold == 0.7
    assert matcher.title_boost_weight == 1.0
    assert matcher.author_boost_weight == 1.0


def test_title_author_matcher_custom_params():
    """Test TitleAuthorMatcher with custom parameters."""
    matcher = TitleAuthorMatcher(
        title_threshold=0.6,
        author_threshold=0.8,
        title_boost_weight=2.0,
        author_boost_weight=1.5
    )
    assert matcher.title_threshold == 0.6
    assert matcher.author_threshold == 0.8
    assert matcher.title_boost_weight == 2.0
    assert matcher.author_boost_weight == 1.5


def test_match_and_boost_no_targets():
    """Test matching with no target title or authors returns unchanged results."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    boosted = matcher.match_and_boost(results, target_title=None, target_authors=None)
    assert boosted == results


def test_match_and_boost_empty_results():
    """Test matching empty results list."""
    matcher = TitleAuthorMatcher()
    results = matcher.match_and_boost([], target_title="attention is all you need")
    assert results == []


def test_match_and_boost_title_exact_match():
    """Test boosting with exact title match."""
    matcher = TitleAuthorMatcher()
    results = [
        SearchResult(doc_id="1", score=0.9, rank=1),
        SearchResult(doc_id="2", score=0.8, rank=2)
    ]
    results[0].title = "Attention Is All You Need"
    results[1].title = "BERT: Pre-training of Deep Bidirectional Transformers"
    
    boosted = matcher.match_and_boost(results, target_title="attention is all you need")
    assert len(boosted) == 2
    # First result should get title boost (score changed)
    assert boosted[0].score != 0.9


def test_match_and_boost_title_partial_match():
    """Test boosting with partial title match."""
    matcher = TitleAuthorMatcher(title_threshold=0.3)
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].title = "Deep Residual Learning for Image Recognition"
    
    boosted = matcher.match_and_boost(results, target_title="residual learning")
    assert len(boosted) == 1


def test_match_and_boost_author_match():
    """Test boosting with author match."""
    matcher = TitleAuthorMatcher()
    results = [
        SearchResult(doc_id="1", score=0.9, rank=1),
        SearchResult(doc_id="2", score=0.8, rank=2)
    ]
    results[0].authors = ["Ashish Vaswani", "Noam Shazeer"]
    results[1].authors = ["Jacob Devlin", "Ming-Wei Chang"]
    
    boosted = matcher.match_and_boost(results, target_authors=["Vaswani"])
    assert len(boosted) == 2


def test_match_and_boost_multiple_authors():
    """Test matching with multiple target authors."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].authors = ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio"]
    
    boosted = matcher.match_and_boost(
        results,
        target_authors=["Hinton", "LeCun"]
    )
    assert len(boosted) == 1


def test_match_and_boost_title_and_author():
    """Test boosting with both title and author matches."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].title = "Attention Is All You Need"
    results[0].authors = ["Ashish Vaswani"]
    
    boosted = matcher.match_and_boost(
        results,
        target_title="attention mechanism",
        target_authors=["Vaswani"]
    )
    assert len(boosted) == 1


def test_match_and_boost_case_insensitive():
    """Test that matching is case-insensitive."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].title = "BERT: PRE-TRAINING OF TRANSFORMERS"
    
    boosted = matcher.match_and_boost(results, target_title="bert pre-training")
    assert len(boosted) == 1


def test_match_and_boost_no_title_attribute():
    """Test matching when results don't have title attribute."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    # No title attribute set
    boosted = matcher.match_and_boost(results, target_title="some title")
    assert len(boosted) == 1


def test_match_and_boost_no_authors_attribute():
    """Test matching when results don't have authors attribute."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    # No authors attribute set
    boosted = matcher.match_and_boost(results, target_authors=["Smith"])
    assert len(boosted) == 1


def test_match_and_boost_updates_ranks():
    """Test that matching updates result ranks."""
    matcher = TitleAuthorMatcher()
    results = [
        SearchResult(doc_id="1", score=0.7, rank=1),
        SearchResult(doc_id="2", score=0.9, rank=2)
    ]
    results[0].title = "Attention Is All You Need"
    results[1].title = "Unrelated Paper"
    
    boosted = matcher.match_and_boost(results, target_title="attention is all you need")
    # Ranks should be updated
    assert boosted[0].rank == 1
    assert boosted[1].rank == 2


def test_match_and_boost_empty_title():
    """Test matching when result has empty title."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].title = ""
    
    boosted = matcher.match_and_boost(results, target_title="some title")
    assert len(boosted) == 1


def test_match_and_boost_empty_authors():
    """Test matching when result has empty authors list."""
    matcher = TitleAuthorMatcher()
    results = [SearchResult(doc_id="1", score=0.9, rank=1)]
    results[0].authors = []
    
    boosted = matcher.match_and_boost(results, target_authors=["Smith"])
    assert len(boosted) == 1


def test_jaccard_similarity_empty_strings():
    """Test Jaccard similarity with empty strings."""
    matcher = TitleAuthorMatcher()
    similarity = matcher._jaccard_similarity("", "some text")
    assert similarity == 0.0
    
    similarity2 = matcher._jaccard_similarity("some text", "")
    assert similarity2 == 0.0


def test_jaccard_similarity_tokens_empty_sets():
    """Test Jaccard similarity with empty token sets."""
    matcher = TitleAuthorMatcher()
    similarity = matcher._jaccard_similarity_tokens(set(), {"a", "b"})
    assert similarity == 0.0
    
    similarity2 = matcher._jaccard_similarity_tokens({"a"}, set())
    assert similarity2 == 0.0


def test_parse_authors_comma_separated():
    """Test parsing authors from comma-separated string."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors("John Smith, Jane Doe, Bob Lee")
    assert len(authors) == 3
    assert "John Smith" in authors
    assert "Jane Doe" in authors


def test_parse_authors_semicolon_separated():
    """Test parsing authors from semicolon-separated string."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors("Alice Brown; Charlie Green")
    assert len(authors) == 2
    assert "Alice Brown" in authors


def test_parse_authors_and_separated():
    """Test parsing authors from 'and' separated string."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors("David White and Emma Black")
    assert len(authors) == 2
    assert "David White" in authors


def test_parse_authors_single_string():
    """Test parsing single author string."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors("Frank Moore")
    assert len(authors) == 1
    assert authors[0] == "Frank Moore"


def test_parse_authors_list_of_strings():
    """Test parsing authors from list of strings."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors(["Grace Lee", "Henry Kim"])
    assert len(authors) == 2
    assert "Grace Lee" in authors


def test_parse_authors_list_of_dicts():
    """Test parsing authors from list of dicts."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors([
        {"name": "Irene Chen"},
        {"full_name": "Jack Wilson"},
        {"author": "Kate Brown"}
    ])
    assert len(authors) == 3
    assert "Irene Chen" in authors
    assert "Jack Wilson" in authors
    assert "Kate Brown" in authors


def test_parse_authors_list_of_dicts_no_name():
    """Test parsing authors from dicts without name field."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors([{"title": "Professor"}, {"id": 123}])
    assert len(authors) == 0


def test_parse_authors_empty_list():
    """Test parsing empty author list."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors([])
    assert len(authors) == 0


def test_parse_authors_invalid_type():
    """Test parsing authors with invalid type."""
    matcher = TitleAuthorMatcher()
    authors = matcher._parse_authors(123)  # Invalid type
    assert len(authors) == 0


def test_get_match_summary_no_match():
    """Test match summary with no boost components."""
    matcher = TitleAuthorMatcher()
    result = SearchResult(doc_id="1", score=0.9, rank=1)
    summary = matcher.get_match_summary(result)
    assert summary is None


def test_get_match_summary_no_matches():
    """Test match summary when boost components exist but no matches."""
    matcher = TitleAuthorMatcher()
    result = SearchResult(doc_id="1", score=0.9, rank=1)
    result.boost_components = {
        'title_match': False,
        'author_match': False,
        'match_boost': 0.0
    }
    summary = matcher.get_match_summary(result)
    assert summary is None


def test_get_match_summary_title_match_only():
    """Test match summary with title match only."""
    matcher = TitleAuthorMatcher()
    result = SearchResult(doc_id="1", score=0.9, rank=1)
    result.boost_components = {
        'title_match': True,
        'title_score': 0.95,
        'author_match': False,
        'match_boost': 1.0
    }
    summary = matcher.get_match_summary(result)
    assert "Title match: 0.95" in summary
    assert "Match boost: +1.00" in summary


def test_get_match_summary_author_match_only():
    """Test match summary with author match only."""
    matcher = TitleAuthorMatcher()
    result = SearchResult(doc_id="1", score=0.9, rank=1)
    result.boost_components = {
        'title_match': False,
        'author_match': True,
        'author_score': 0.88,
        'match_boost': 1.0
    }
    summary = matcher.get_match_summary(result)
    assert "Author match: 0.88" in summary


def test_get_match_summary_both_matches():
    """Test match summary with both title and author matches."""
    matcher = TitleAuthorMatcher()
    result = SearchResult(doc_id="1", score=0.9, rank=1)
    result.boost_components = {
        'title_match': True,
        'title_score': 0.92,
        'author_match': True,
        'author_score': 0.85,
        'match_boost': 2.0
    }
    summary = matcher.get_match_summary(result)
    assert "Title match: 0.92" in summary
    assert "Author match: 0.85" in summary
    assert "Match boost: +2.00" in summary


def test_normalize_string_punctuation_removal():
    """Test string normalization removes punctuation."""
    matcher = TitleAuthorMatcher()
    normalized = matcher._normalize_string("Hello, World! How's it going?")
    assert normalized == "hello world hows it going"


def test_normalize_string_multiple_spaces():
    """Test string normalization collapses multiple spaces."""
    matcher = TitleAuthorMatcher()
    normalized = matcher._normalize_string("too    many     spaces")
    assert normalized == "too many spaces"
