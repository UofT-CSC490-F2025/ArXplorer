import pytest

from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.rerankers.jina_reranker import JinaReranker
from src.retrieval.rerankers.qwen_reranker import QwenReranker
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


def test_qwen_reranker(monkeypatch):
    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0

        @classmethod
        def from_pretrained(cls, model_name, padding_side="left"):
            return cls()

        def convert_tokens_to_ids(self, token):
            return 1 if token == "yes" else 0

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def __call__(self, texts, **kwargs):
            fake = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

            class FakeTokens(dict):
                def to(self_inner, device):
                    return self_inner

            return FakeTokens(fake)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            class P:
                def numel(self_inner):
                    return 0
            return [P()]

    monkeypatch.setattr("src.retrieval.rerankers.qwen_reranker.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("src.retrieval.rerankers.qwen_reranker.AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(QwenReranker, "_process_inputs", lambda self, pairs: {"pairs": pairs})
    monkeypatch.setattr(QwenReranker, "_compute_scores", lambda self, inputs: [float(len(p)) for p in inputs["pairs"]])

    reranker = QwenReranker(DOC_TEXTS, model_name="fake-qwen", device="cpu", batch_size=2)
    results = reranker.rerank("query", make_results(), top_k=2)
    assert [r.doc_id for r in results] == ["doc2", "doc1"]

    # No valid texts -> returns originals
    reranker_missing = QwenReranker({}, model_name="fake-qwen", device="cpu")
    base = make_results()
    out = reranker_missing.rerank("query", base, top_k=1)
    assert out[0].doc_id == base[0].doc_id


def test_qwen_reranker_full_pipeline(monkeypatch):
    import numpy as np
    import types

    class FakeTensor:
        def __init__(self, data):
            self.array = np.array(data, dtype=np.float32)

        def __getitem__(self, idx):
            res = self.array[idx]
            if isinstance(res, np.ndarray):
                return FakeTensor(res)
            return FakeTensor(res)

        def __len__(self):
            return len(self.array)

        def exp(self):
            return FakeTensor(np.exp(self.array))

        def tolist(self):
            return self.array.tolist()

    class FakeFunctional:
        @staticmethod
        def log_softmax(tensor, dim):
            arr = tensor.array
            shifted = arr - np.max(arr, axis=dim, keepdims=True)
            logits = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
            return FakeTensor(logits)

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        nn = types.SimpleNamespace(functional=FakeFunctional())

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        cuda = _Cuda()

        @staticmethod
        def tensor(data, device=None):
            return FakeTensor(data)

        @staticmethod
        def stack(tensors, dim=0):
            arrays = [
                t.array if isinstance(t, FakeTensor) else np.array(t, dtype=np.float32)
                for t in tensors
            ]
            return FakeTensor(np.stack(arrays, axis=dim))

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __call__(self, fn):
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)

                return wrapper

        @staticmethod
        def no_grad():
            return FakeTorch._NoGrad()

    class FakeTokenizer:
        def __init__(self, padding_side="left"):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self.padding_side = padding_side

        @classmethod
        def from_pretrained(cls, model_name, padding_side="left"):
            return cls(padding_side=padding_side)

        def convert_tokens_to_ids(self, token):
            return 1 if token == "yes" else 0

        def encode(self, text, add_special_tokens=False):
            return [len(text) % 5, (len(text) + 1) % 5]

        def __call__(self, pairs, **kwargs):
            ids = [[(idx + 1) for idx, _ in enumerate(pair)] for pair in pairs]
            return {"input_ids": ids}

    class FakeModel:
        def __init__(self):
            self.calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return [types.SimpleNamespace(numel=lambda: 10)]

        def __call__(self, **inputs):
            batch = inputs["input_ids"].array.shape[0]
            logits = []
            for i in range(batch):
                yes = 1.0 + 0.5 * (self.calls + i)
                no = 0.2
                logits.append([[no, yes, 0.0]])
            self.calls += batch
            return types.SimpleNamespace(logits=FakeTensor(logits))

    monkeypatch.setattr("src.retrieval.rerankers.qwen_reranker.torch", FakeTorch)
    monkeypatch.setattr(
        "src.retrieval.rerankers.qwen_reranker.AutoTokenizer", FakeTokenizer
    )
    monkeypatch.setattr(
        "src.retrieval.rerankers.qwen_reranker.AutoModelForCausalLM", FakeModel
    )

    reranker = QwenReranker(DOC_TEXTS, model_name="fake-qwen", device=None, batch_size=1)
    results = reranker.rerank("query", make_results(), top_k=2)
    assert [r.doc_id for r in results] == ["doc2", "doc1"]
    assert reranker.token_true_id == 1
    assert reranker.token_false_id == 0

    # Explicit CUDA path to exercise FP16 logging
    reranker_cuda = QwenReranker(DOC_TEXTS, model_name="fake-qwen", device="cuda", batch_size=1)
    assert reranker_cuda.device == "cuda"


def test_qwen_reranker_empty_results(monkeypatch):
    class TinyTokenizer:
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, model_name, padding_side="left"):
            return cls()

        def convert_tokens_to_ids(self, token):
            return 0

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2]]}

    class TinyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr("src.retrieval.rerankers.qwen_reranker.AutoTokenizer", TinyTokenizer)
    monkeypatch.setattr("src.retrieval.rerankers.qwen_reranker.AutoModelForCausalLM", TinyModel)
    monkeypatch.setattr(QwenReranker, "_process_inputs", lambda self, pairs: {"pairs": pairs})
    monkeypatch.setattr(QwenReranker, "_compute_scores", lambda self, inputs: [])

    reranker = QwenReranker(DOC_TEXTS, model_name="fake-qwen", device="cpu")
    assert reranker.rerank("query", [], top_k=3) == []
