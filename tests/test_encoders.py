import numpy as np
import pytest
import sys
import types

from src.retrieval.encoders.dense import DenseEncoder
from src.retrieval.encoders.sparse import SparseEncoder


def test_dense_encoder_sentence_transformer(monkeypatch):
    def fake_load(self, model_name):
        class FakeModel:
            def encode(self, texts, batch_size=16, device=None, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
                arr = np.array([[len(t), len(t) + 1, len(t) + 2] for t in texts], dtype=np.float32)
                if normalize_embeddings:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    arr = arr / (norms + 1e-8)
                return arr

            def get_sentence_embedding_dimension(self):
                return 3

        self.model = FakeModel()
        self._dimension = self.model.get_sentence_embedding_dimension()

    monkeypatch.setattr(DenseEncoder, "_load_sentence_transformer", fake_load)

    encoder = DenseEncoder(model_name="dummy", device="cpu", normalize=True)
    vecs = encoder.encode(["abc", "de"])

    assert vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert encoder.get_dimension() == 3
    assert encoder.model_name == "dummy"


def test_dense_encoder_sentence_transformer_full_stack(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            self._dim = 4
            self._device = device

        def get_sentence_embedding_dimension(self):
            return self._dim

        def encode(
            self,
            texts,
            batch_size=16,
            device=None,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ):
            arr = np.array(
                [[len(t), len(t) + 1, len(t) + 2, len(t) + 3] for t in texts],
                dtype=np.float32,
            )
            if normalize_embeddings:
                arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            return arr

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    encoder = DenseEncoder(model_name="fake-specter", normalize=False)
    vectors = encoder.encode("abc", batch_size=1)

    assert vectors.shape == (1, 4)
    assert encoder.tokenizer is None
    assert encoder.device == "cpu"


def test_dense_encoder_specter2(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.adapters_loaded = {}
            self.active_adapters = None

        def load_adapter(self, adapter_name, source="hf", set_active=False):
            handle = f"{adapter_name}_handle"
            self.adapters_loaded[adapter_name] = handle
            return handle

        def set_active_adapters(self, adapter_handle):
            self.active_adapters = adapter_handle

        def to(self, device):
            return self

    def fake_load_specter2(self, adapter_name):
        self.model = FakeModel()
        self.tokenizer = object()
        self.loaded_adapters = {adapter_name: adapter_name}
        self.current_adapter = adapter_name
        self._dimension = 5

    def fake_encode_specter2(self, texts, batch_size):
        return np.full((len(texts), self._dimension), 0.5, dtype=np.float32)

    monkeypatch.setattr(DenseEncoder, "_load_specter2", fake_load_specter2)
    monkeypatch.setattr(DenseEncoder, "_encode_specter2", fake_encode_specter2)

    encoder = DenseEncoder(model_name="dummy", device="cpu", use_specter2=True)
    encoder.use_query_adapter()
    assert encoder.current_adapter == encoder.specter2_query_adapter
    encoder.use_base_adapter()
    assert encoder.current_adapter == encoder.specter2_base_adapter

    vecs = encoder.encode(["hello"], batch_size=1)
    assert vecs.shape == (1, 5)
    assert np.all(vecs == 0.5)


def test_sparse_encoder(monkeypatch):
    def fake_encode_batch(self, texts):
        return [
            (np.array([0, len(text)], dtype=np.int64), np.array([1.0, float(len(text))], dtype=np.float32))
            for text in texts
        ]

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return object()

    class FakeModel:
        def __init__(self):
            self.config = type("cfg", (), {"vocab_size": 999})

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(SparseEncoder, "_encode_batch", fake_encode_batch)
    monkeypatch.setattr("src.retrieval.encoders.sparse.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("src.retrieval.encoders.sparse.AutoModelForMaskedLM", FakeModel)

    encoder = SparseEncoder(model_name="dummy", device=None)
    vectors = encoder.encode(["abc", "de"], batch_size=1)

    assert len(vectors) == 2
    indices, values = vectors[0]
    assert np.array_equal(indices, np.array([0, 3]))
    assert np.array_equal(values, np.array([1.0, 3.0], dtype=np.float32))
    assert encoder.get_dimension() == 999


def test_dense_encoder_specter2_full_path(monkeypatch):
    class FakeTensor:
        def __init__(self, array):
            self.array = np.array(array)

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def __getitem__(self, idx):
            res = self.array[idx]
            if isinstance(res, np.ndarray):
                return FakeTensor(res)
            return res

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, texts, **kwargs):
            batch = len(texts)
            return {
                "input_ids": FakeTensor(np.ones((batch, 2, 4))),
                "attention_mask": FakeTensor(np.ones((batch, 2))),
            }

    class FakeAdapterModel:
        def __init__(self):
            self.adapters_loaded = {}
            self.active_adapters = None

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_adapter(self, adapter_name, source="hf", set_active=False):
            handle = f"{adapter_name}_handle"
            self.adapters_loaded[adapter_name] = handle
            return handle

        def set_active_adapters(self, adapter_handle):
            self.active_adapters = adapter_handle

        def __call__(self, **inputs):
            batch = len(inputs["input_ids"].array)
            data = np.full((batch, 1, 4), 2.0, dtype=np.float32)
            return types.SimpleNamespace(last_hidden_state=FakeTensor(data))

    fake_adapters = types.ModuleType("adapters")
    fake_adapters.AutoAdapterModel = FakeAdapterModel
    monkeypatch.setitem(sys.modules, "adapters", fake_adapters)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    encoder = DenseEncoder(model_name="dummy", device="cpu", use_specter2=True, normalize=True)
    vecs = encoder.encode(["hello", "world"], batch_size=2)
    assert vecs.shape == (2, 4)
    encoder.use_query_adapter()
    encoder.use_base_adapter()
    # Calling set_adapter with the current adapter exercises the early return branch (line 162)
    encoder.set_adapter(encoder.specter2_base_adapter)
    assert encoder.current_adapter == encoder.specter2_base_adapter


def test_dense_encoder_set_adapter_warning(monkeypatch, capsys):
    monkeypatch.setattr(
        DenseEncoder,
        "_load_sentence_transformer",
        lambda self, model_name: setattr(self, "_dimension", 4) or setattr(self, "model", object()),
    )
    encoder = DenseEncoder(model_name="dummy", device="cpu", use_specter2=False)
    encoder.set_adapter("foo")
    assert "set_adapter called but use_specter2=False" in capsys.readouterr().out


def test_dense_encoder_requires_adapters(monkeypatch):
    monkeypatch.delitem(sys.modules, "adapters", raising=False)
    with pytest.raises(ImportError):
        DenseEncoder(model_name="dummy", use_specter2=True)


def test_sparse_encoder_full_batch_execution(monkeypatch):
    class FakeTensor:
        def __init__(self, array):
            self.array = np.array(array)

        def to(self, _device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.array, axis))

        def max(self, dim):
            values = np.max(self.array, axis=dim)
            return FakeTensor(values), None

        def __mul__(self, other):
            other_arr = other.array if isinstance(other, FakeTensor) else other
            return FakeTensor(self.array * other_arr)

        def __add__(self, other):
            other_arr = other.array if isinstance(other, FakeTensor) else other
            return FakeTensor(self.array + other_arr)

        __radd__ = __add__

        def relu(self):
            return FakeTensor(np.maximum(self.array, 0))

        def log(self):
            return FakeTensor(np.log(self.array))

        def __iter__(self):
            for row in self.array:
                if isinstance(row, np.ndarray):
                    yield FakeTensor(row)
                else:
                    yield row

        def nonzero(self):
            return FakeTensor(np.flatnonzero(self.array))

        def squeeze(self):
            return FakeTensor(np.squeeze(self.array))

        def __getitem__(self, idx):
            res = self.array[idx]
            if isinstance(res, np.ndarray):
                return FakeTensor(res)
            return res

    class FakeTokens(dict):
        def __init__(self, batch, seq_len):
            input_ids = FakeTensor(np.ones((batch, seq_len)))
            attention = FakeTensor(np.ones((batch, seq_len)))
            super().__init__({"input_ids": input_ids, "attention_mask": attention})
            self.attention_mask = attention

        def to(self, _device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __call__(self, texts, **_kwargs):
            return FakeTokens(len(texts), 3)

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(vocab_size=4)

        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **_tokens):
            batch = _tokens["input_ids"].array.shape[0]
            seq_len = _tokens["input_ids"].array.shape[1]
            vocab = self.config.vocab_size
            logits = np.zeros((batch, seq_len, vocab), dtype=np.float32)
            logits[:, :, 0] = 1.0  # single non-zero entry to trigger indices.ndim == 0 branch
            return types.SimpleNamespace(logits=FakeTensor(logits))

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        cleared = False

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            FakeCuda.cleared = True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(name):
            return types.SimpleNamespace(type=name)

        @staticmethod
        def no_grad():
            return FakeNoGrad()

        @staticmethod
        def relu(tensor):
            return tensor.relu()

        @staticmethod
        def log(tensor):
            return FakeTensor(np.log(tensor.array))

    monkeypatch.setattr("src.retrieval.encoders.sparse.torch", FakeTorch)
    monkeypatch.setattr("src.retrieval.encoders.sparse.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("src.retrieval.encoders.sparse.AutoModelForMaskedLM", FakeModel)

    FakeCuda.cleared = False
    encoder = SparseEncoder(model_name="fake-sparse", device="cuda", max_length=8)
    vectors = encoder.encode("single input", batch_size=1)

    assert len(vectors) == 1
    indices, values = vectors[0]
    assert indices.size > 0
    assert values.size == indices.size
    assert encoder.device.type == "cuda"
    assert FakeCuda.cleared is True
    assert encoder.model_name == "fake-sparse"
