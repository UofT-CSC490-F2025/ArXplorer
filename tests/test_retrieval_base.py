from pathlib import Path
from typing import List
import shutil

import pytest

from src.data.document import Document
from src.retrieval.encoders.base import BaseEncoder
from src.retrieval.indexers.base import BaseIndexer
from src.retrieval.searchers.base import BaseSearcher, SearchResult
from src.retrieval.rerankers.base import BaseReranker


TMP_DIR = Path(__file__).parent / "tmp_data" / "retrieval_base"
TMP_DIR.mkdir(parents=True, exist_ok=True)


class DummyEncoder(BaseEncoder):
    def __init__(self, name: str = "dummy-encoder"):
        self._name = name
        self._dim = 4

    def encode(self, texts, batch_size: int = 16):
        if isinstance(texts, str):
            texts = [texts]
        return [[len(t)] * self._dim for t in texts]

    def get_dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name


class DummyIndexer(BaseIndexer):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.added: List[Document] = []
        self.saved = False
        self.loaded = False

    def add_documents(self, documents: List[Document], batch_size: int = 16, show_progress: bool = True) -> None:
        self.added.extend(documents)

    def save(self) -> None:
        self.saved = True

    def load(self) -> None:
        self.loaded = True

    def get_num_documents(self) -> int:
        return len(self.added)


class DummySearcher(BaseSearcher):
    def __init__(self):
        self.queries = []

    def search(self, query: str, top_k: int = 10):
        self.queries.append((query, top_k))
        return [
            SearchResult(doc_id="doc1", score=1.0, rank=1),
            SearchResult(doc_id="doc2", score=0.5, rank=2),
        ][:top_k]


class DummyReranker(BaseReranker):
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 10):
        reranked = list(reversed(results))[:top_k]
        for idx, result in enumerate(reranked, start=1):
            result.rank = idx
            result.cross_encoder_score = (result.cross_encoder_score or result.score) + 0.1
        return reranked


def test_dummy_encoder_behaves_like_base():
    encoder = DummyEncoder(name="custom")
    vectors = encoder.encode(["abc", "de"])
    assert vectors == [[3, 3, 3, 3], [2, 2, 2, 2]]
    assert encoder.get_dimension() == 4
    assert encoder.model_name == "custom"


def test_dummy_indexer_tracks_documents():
    idx_dir = TMP_DIR / "index"
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    indexer = DummyIndexer(str(idx_dir))
    docs = [Document(id="1", text="hello"), Document(id="2", text="world")]

    indexer.add_documents(docs)
    assert indexer.get_num_documents() == 2
    assert [d.id for d in indexer.added] == ["1", "2"]

    indexer.save()
    indexer.load()
    assert indexer.saved is True
    assert indexer.loaded is True
    assert idx_dir.exists()


def test_dummy_searcher_returns_results():
    searcher = DummySearcher()
    results = searcher.search("neural nets", top_k=1)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, SearchResult)
    assert result.doc_id == "doc1"
    assert result.rank == 1
    assert searcher.queries == [("neural nets", 1)]


def test_dummy_reranker_updates_ranks_and_scores():
    reranker = DummyReranker()
    base_results = [
        SearchResult(doc_id="doc1", score=0.2, rank=1),
        SearchResult(doc_id="doc2", score=0.9, rank=2),
    ]

    new_results = reranker.rerank("query", base_results, top_k=2)
    assert [r.doc_id for r in new_results] == ["doc2", "doc1"]
    assert [r.rank for r in new_results] == [1, 2]
    assert new_results[0].cross_encoder_score == pytest.approx(1.0)


def test_base_methods_can_be_called_directly_for_coverage():
    encoder = DummyEncoder()
    BaseEncoder.encode(encoder, "text")
    BaseEncoder.get_dimension(encoder)
    BaseEncoder.model_name.fget(encoder)

    indexer = DummyIndexer(str(TMP_DIR / "base_calls"))
    BaseIndexer.add_documents(indexer, [])
    BaseIndexer.save(indexer)
    BaseIndexer.load(indexer)
    BaseIndexer.get_num_documents(indexer)

    searcher = DummySearcher()
    BaseSearcher.search(searcher, "query")

    reranker = DummyReranker()
    BaseReranker.rerank(reranker, "query", [], top_k=1)
