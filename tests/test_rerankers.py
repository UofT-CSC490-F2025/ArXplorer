import pytest

from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.rerankers.jina_reranker import JinaReranker
from src.retrieval.searchers.base import SearchResult


class FakeCrossEncoder:
    def __init__(self, model_name, device=None, max_length=512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

    def predict(self, pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        # generate deterministic scores: length of doc text
        return [len(doc) for _, doc in pairs]


DOC_TEXTS = {"doc1": "short text", "doc2": "much longer text for testing"}


def make_results():
    res1 = SearchResult(doc_id="doc1", score=0.1, dense_score=0.2, sparse_score=0.3)
    res1.title = "Title1"
    res1.abstract = "Abstract1"
    res1.authors = ["Alice"]
    res1.categories = ["cs.AI"]

    res2 = SearchResult(doc_id="doc2", score=0.2, dense_score=0.5, sparse_score=0.6)
    res2.citation_count = 10
    res2.publication_year = 2022
    return [res1, res2]


def test_cross_encoder_reranker_sorts_and_preserves(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = CrossEncoderReranker(DOC_TEXTS, model_name="fake-model", device="cpu")
    results = reranker.rerank("query", make_results(), top_k=2)

    assert [r.doc_id for r in results] == ["doc2", "doc1"]
    assert results[0].cross_encoder_score == len(DOC_TEXTS["doc2"])
    assert results[0].dense_score == make_results()[1].dense_score
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert hasattr(results[1], "title")


def test_cross_encoder_reranker_handles_missing_text(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.retrieval.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = CrossEncoderReranker({"doc1": "text only"}, model_name="fake-model", device="cpu")
    results = reranker.rerank("query", make_results(), top_k=2)

    # Only doc1 has text; doc2 should be skipped and result list limited
    assert [r.doc_id for r in results] == ["doc1"]
    out = capsys.readouterr().out
    assert "No text found for doc_id doc2" in out


def test_cross_encoder_reranker_no_pairs_returns_original(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = CrossEncoderReranker({}, model_name="fake-model", device="cpu")
    base = make_results()
    results = reranker.rerank("query", base, top_k=1)

    # Nothing rewritable -> returns original top slice
    assert results[0].doc_id == base[0].doc_id


def test_cross_encoder_auto_device(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = CrossEncoderReranker(DOC_TEXTS, model_name="fake-model", device=None)
    assert reranker.device == "cpu"


def test_cross_encoder_empty_results(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.rerankers.cross_encoder_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = CrossEncoderReranker(DOC_TEXTS, model_name="fake-model", device="cpu")
    assert reranker.rerank("query", [], top_k=5) == []


def test_jina_reranker(monkeypatch, capsys):
    class FakeJinaModel:
        def __init__(self):
            self._called_docs = []

        def rerank(self, query, docs):
            self._called_docs.extend(docs)
            return [{"document": d, "relevance_score": float(len(d))} for d in docs]

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr(
        "src.retrieval.rerankers.jina_reranker.AutoModel",
        type("X", (), {"from_pretrained": staticmethod(lambda *a, **k: FakeJinaModel())}),
    )

    reranker = JinaReranker(DOC_TEXTS, model_name="fake-jina", device="cpu")
    results = reranker.rerank("query", make_results(), top_k=1)
    assert [r.doc_id for r in results] == ["doc2"]

    # Missing doc text case
    reranker_missing = JinaReranker({"doc1": "text"}, model_name="fake-jina", device="cpu")
    out = reranker_missing.rerank("query", make_results(), top_k=2)
    assert [r.doc_id for r in out] == ["doc1"]
    assert "No text found for doc_id doc2" in capsys.readouterr().out


def test_jina_reranker_auto_device(monkeypatch):
    class TinyModel:
        def rerank(self, query, docs):
            return [{"document": d, "relevance_score": 1.0} for d in docs]

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr(
        "src.retrieval.rerankers.jina_reranker.AutoModel",
        type("X", (), {"from_pretrained": staticmethod(lambda *a, **k: TinyModel())}),
    )

    reranker = JinaReranker(DOC_TEXTS, model_name="tiny", device=None)
    assert reranker.device == "cpu"
    assert reranker.rerank("q", [], top_k=2) == []


def test_jina_reranker_gpu_path(monkeypatch):
    class GPUAwareModel:
        def __init__(self):
            self.moved_to = None

        def rerank(self, query, docs):
            return [{"document": d, "relevance_score": 1.0} for d in docs]

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    fake = GPUAwareModel()
    monkeypatch.setattr(
        "src.retrieval.rerankers.jina_reranker.AutoModel",
        type("X", (), {"from_pretrained": staticmethod(lambda *a, **k: fake)}),
    )

    reranker = JinaReranker(DOC_TEXTS, model_name="tiny", device="cuda")
    assert fake.moved_to == "cuda"


def test_jina_reranker_no_documents(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.retrieval.rerankers.jina_reranker.AutoModel",
        type(
            "X",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda *a, **k: type(
                        "Y",
                        (),
                        {
                            "rerank": staticmethod(lambda q, d: []),
                            "to": staticmethod(lambda self=None, device=None: self),
                            "eval": staticmethod(lambda self=None: self),
                            "parameters": staticmethod(lambda: []),
                        },
                    )()
                )
            },
        ),
    )

    reranker = JinaReranker({}, model_name="tiny", device="cpu")
    base = make_results()
    out = reranker.rerank("q", base, top_k=1)
    assert out[0].doc_id == base[0].doc_id


