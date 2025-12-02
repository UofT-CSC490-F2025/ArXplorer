import sys
from pathlib import Path
import types
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Add project root to path so 'from src.' imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_stub_modules():
    # Torch stub (used during import but not instantiated in tests)
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        torch_stub.cuda = _Cuda()

        def device(name=None):
            return name
        torch_stub.device = device
        class _Tensor(list):
            pass

        torch_stub.Tensor = _Tensor

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __call__(self, func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

                return wrapper

        def no_grad():
            return _NoGrad()

        torch_stub.no_grad = no_grad
        torch_stub.float32 = "float32"
        torch_stub.float16 = "float16"
        sys.modules["torch"] = torch_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class _FakeTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class _FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(vocab_size=0)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

            def parameters(self):
                return []

        class _FakeRerankModel(_FakeModel):
            def rerank(self, query, docs):
                return [{"document": d, "relevance_score": float(len(d))} for d in docs]

        transformers_stub.AutoTokenizer = _FakeTokenizer
        transformers_stub.AutoModelForMaskedLM = _FakeModel
        transformers_stub.AutoModelForCausalLM = _FakeModel
        transformers_stub.AutoModelForSeq2SeqLM = _FakeModel
        transformers_stub.AutoModel = _FakeRerankModel
        sys.modules["transformers"] = transformers_stub

    if "sentence_transformers" not in sys.modules:
        sentence_stub = types.ModuleType("sentence_transformers")

        class _SentenceTransformer:
            def __init__(self, *args, **kwargs):
                self._dim = 768

            def get_sentence_embedding_dimension(self):
                return self._dim

            def __call__(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [[0.0] * self._dim for _ in texts]

        class _CrossEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs):
                return [0.0 for _ in pairs]

        sentence_stub.SentenceTransformer = _SentenceTransformer
        sentence_stub.CrossEncoder = _CrossEncoder
        sys.modules["sentence_transformers"] = sentence_stub

    if "pymilvus" not in sys.modules:
        pymilvus_stub = types.ModuleType("pymilvus")

        class _Connections:
            @staticmethod
            def connect(**kwargs):
                return None

        pymilvus_stub.connections = _Connections()

        class _Utility:
            @staticmethod
            def has_collection(name):
                return False

            @staticmethod
            def list_collections():
                return []

        pymilvus_stub.utility = _Utility()

        class _FieldSchema:
            def __init__(self, *args, **kwargs):
                pass

        class _CollectionSchema:
            def __init__(self, *args, **kwargs):
                pass

        class _AnnSearchRequest:
            def __init__(self, *args, **kwargs):
                pass

        class _RRFRanker:
            def __init__(self, *args, **kwargs):
                pass

        class _Collection:
            def __init__(self, *args, **kwargs):
                self.num_entities = 0

            def load(self):
                return None

            def insert(self, *args, **kwargs):
                return None

            def create_index(self, *args, **kwargs):
                return None

        class _DataType:
            VARCHAR = "varchar"
            JSON = "json"
            INT32 = "int32"
            FLOAT_VECTOR = "float_vector"
            SPARSE_FLOAT_VECTOR = "sparse_float_vector"

        pymilvus_stub.Collection = _Collection
        pymilvus_stub.CollectionSchema = _CollectionSchema
        pymilvus_stub.FieldSchema = _FieldSchema
        pymilvus_stub.DataType = _DataType
        pymilvus_stub.AnnSearchRequest = _AnnSearchRequest
        pymilvus_stub.RRFRanker = _RRFRanker
        sys.modules["pymilvus"] = pymilvus_stub




_ensure_stub_modules()
