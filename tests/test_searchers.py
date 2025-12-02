import numpy as np
import pytest

from src.retrieval.searchers.milvus_hybrid_searcher import MilvusHybridSearcher


class FakeDenseEncoder:
    def encode(self, queries):
        return np.array([[float(len(q)), 0.5] for q in queries], dtype=np.float32)


class FakeSparseEncoder:
    def encode(self, queries, batch_size=1):
        outputs = []
        for q in queries:
            length = max(1, len(q))
            indices = np.arange(length, dtype=np.int64)
            values = np.linspace(1.0, 0.5, num=length, dtype=np.float32)
            outputs.append((indices, values))
        return outputs


class FakeAnnSearchRequest:
    def __init__(self, data, anns_field, param, limit, expr=None):
        self.data = data
        self.anns_field = anns_field
        self.param = param
        self.limit = limit
        self.expr = expr


class FakeRRFRanker:
    def __init__(self, k):
        self.k = k


class FakeEntity(dict):
    pass


class FakeHit:
    def __init__(self, doc_id, score, year=2023, citations=100):
        self.score = score
        self.entity = FakeEntity(
            {
                "id": doc_id,
                "title": f"title-{doc_id}",
                "abstract": f"abstract-{doc_id}",
                "authors": ["a1", "a2"],
                "categories": ["cs.AI"],
                "year": year,
                "citation_count": citations,
            }
        )


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.loaded = False
        self.num_entities = 42
        self.last_hybrid_call = None
        self.last_query = None

    def load(self):
        self.loaded = True

    def hybrid_search(self, reqs, rerank, limit, output_fields):
        self.last_hybrid_call = {
            "reqs": reqs,
            "rerank": rerank,
            "limit": limit,
            "fields": output_fields,
        }
        hits = [FakeHit("doc-1", 0.9), FakeHit("doc-2", 0.8)]
        return [hits[:limit]]

    def query(self, expr, output_fields):
        self.last_query = {"expr": expr, "fields": output_fields}
        return [
            {
                "id": "doc-1",
                "title": "Doc 1",
                "abstract": "abs",
                "authors": ["a1"],
                "categories": ["cs.AI"],
                "year": 2024,
                "citation_count": 50,
            }
        ]


@pytest.fixture
def patched_milvus(monkeypatch):
    fake_collection = FakeCollection("arxplorer_papers")

    def fake_connect(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "src.retrieval.searchers.milvus_hybrid_searcher.connections.connect",
        fake_connect,
    )
    monkeypatch.setattr(
        "src.retrieval.searchers.milvus_hybrid_searcher.Collection", lambda name: fake_collection
    )
    monkeypatch.setattr(
        "src.retrieval.searchers.milvus_hybrid_searcher.AnnSearchRequest",
        FakeAnnSearchRequest,
    )
    monkeypatch.setattr(
        "src.retrieval.searchers.milvus_hybrid_searcher.RRFRanker", FakeRRFRanker
    )
    return fake_collection


def test_hybrid_search_builds_requests_and_returns_results(patched_milvus):
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        host="fake",
        port=1234,
        collection_name="arxplorer_papers",
        rrf_k=10,
    )

    results = searcher.search(
        query="cat physics",
        top_k=2,
        retrieval_k=5,
        filter_expr="year >= 2020",
        query_variants=["dog vision"],
    )

    assert len(results) == 2
    assert results[0].doc_id == "doc-1"
    assert results[0].title == "title-doc-1"
    assert results[0].citation_count == 100

    hybrid_call = patched_milvus.last_hybrid_call
    assert hybrid_call["limit"] == 2
    assert isinstance(hybrid_call["rerank"], FakeRRFRanker)
    assert len(hybrid_call["reqs"]) == 4  # dense + sparse per query
    assert {req.expr for req in hybrid_call["reqs"]} == {"year >= 2020"}
    assert {"dense_vector", "sparse_vector"} == {req.anns_field for req in hybrid_call["reqs"]}


def test_search_with_scores_returns_rrf_map(patched_milvus):
    searcher = MilvusHybridSearcher(FakeDenseEncoder(), FakeSparseEncoder())
    results, scores = searcher.search_with_scores("graph nets", top_k=1)

    assert len(results) == 1
    doc_id = results[0].doc_id
    assert scores[doc_id]["rrf_score"] == results[0].score
    assert scores[doc_id]["dense_score"] is None
    assert scores[doc_id]["sparse_score"] is None


def test_get_doc_metadata_queries_collection(patched_milvus):
    searcher = MilvusHybridSearcher(FakeDenseEncoder(), FakeSparseEncoder())
    metadata = searcher.get_doc_metadata(["doc-1", "doc-2"])

    assert patched_milvus.last_query["expr"] == 'id in ["doc-1", "doc-2"]'
    assert metadata[0]["title"] == "Doc 1"


def test_hybrid_searcher_connection_failure(monkeypatch):
    def fail_connect(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(
        "src.retrieval.searchers.milvus_hybrid_searcher.connections.connect", fail_connect
    )
    with pytest.raises(ConnectionError):
        MilvusHybridSearcher(FakeDenseEncoder(), FakeSparseEncoder())


def test_search_multi_query_with_filters(patched_milvus):
    """Test multi-query search with varied filters."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    results = searcher.search_multi_query_with_filters(
        original_query="neural networks",
        rewrites=["deep learning", "artificial neural nets"],
        year_constraint={'min': 2020, 'max': None},
        citation_threshold=100,
        top_k=2,
        retrieval_k=5
    )
    
    assert len(results) == 2
    assert results[0].doc_id == "doc-1"
    
    # Should have created multiple search requests
    hybrid_call = patched_milvus.last_hybrid_call
    assert len(hybrid_call["reqs"]) > 2  # Multiple queries + modalities


def test_search_multi_query_without_filters(patched_milvus):
    """Test multi-query search without any filters."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    results = searcher.search_multi_query_with_filters(
        original_query="transformers",
        rewrites=["attention mechanism"],
        year_constraint=None,
        citation_threshold=None,
        top_k=2,
        retrieval_k=5
    )
    
    assert len(results) == 2


def test_search_multi_query_with_year_only(patched_milvus):
    """Test multi-query search with year constraint only."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    results = searcher.search_multi_query_with_filters(
        original_query="computer vision",
        rewrites=["image recognition"],
        year_constraint={'min': 2018, 'max': 2022},
        citation_threshold=None,
        top_k=1,
        retrieval_k=5
    )
    
    assert len(results) == 1


def test_build_filter_expr_both_constraints(patched_milvus):
    """Test filter expression building with both year and citation constraints."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint={'min': 2019, 'max': 2023},
        citation_threshold=500
    )
    
    assert "year >= 2019" in expr
    assert "year <= 2023" in expr
    assert "citation_count >= 500" in expr
    assert " and " in expr


def test_build_filter_expr_year_min_only(patched_milvus):
    """Test filter expression with only minimum year."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint={'min': 2020, 'max': None},
        citation_threshold=None
    )
    
    assert expr == "year >= 2020"


def test_build_filter_expr_year_max_only(patched_milvus):
    """Test filter expression with only maximum year."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint={'min': None, 'max': 2015},
        citation_threshold=None
    )
    
    assert expr == "year <= 2015"


def test_build_filter_expr_citation_only(patched_milvus):
    """Test filter expression with only citation threshold."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint=None,
        citation_threshold=200
    )
    
    assert expr == "citation_count >= 200"


def test_build_filter_expr_zero_citation(patched_milvus):
    """Test filter expression ignores zero citation threshold."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint=None,
        citation_threshold=0
    )
    
    assert expr is None


def test_build_filter_expr_negative_citation(patched_milvus):
    """Test filter expression ignores negative citation threshold."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint=None,
        citation_threshold=-10
    )
    
    assert expr is None


def test_build_filter_expr_none_constraints(patched_milvus):
    """Test filter expression with no constraints."""
    searcher = MilvusHybridSearcher(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    
    expr = searcher._build_filter_expr(
        year_constraint=None,
        citation_threshold=None
    )
    
    assert expr is None
