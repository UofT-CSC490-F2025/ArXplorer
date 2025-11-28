import pytest

from src.retrieval.query_rewriting.llm_rewriter import (
    build_milvus_filter_expr,
    LLMQueryRewriter,
)
from src.retrieval.query_rewriting.base import BaseQueryRewriter


def test_build_filter_expr_full_filters():
    filters = {
        "year": {"min": 2018, "max": 2022},
        "citation_count": {"min": 100, "max": 1000},
    }
    expr = build_milvus_filter_expr(filters)
    assert expr == "year >= 2018 and year <= 2022 and citation_count >= 100 and citation_count <= 1000"


def test_build_filter_expr_year_only():
    filters = {"year": {"min": 2020, "max": None}}
    expr = build_milvus_filter_expr(filters, enable_citation_filters=False)
    assert expr == "year >= 2020"


def test_build_filter_expr_none_returns_none():
    assert build_milvus_filter_expr({}) is None
    assert build_milvus_filter_expr({"citation_count": {}}) is None


class DummyRewriter(BaseQueryRewriter):
    def rewrite(self, query: str, num_rewrites: int = 1):
        BaseQueryRewriter.rewrite(self, query, num_rewrites)
        return [query.upper()]

    @property
    def model_name(self) -> str:
        BaseQueryRewriter.model_name.__get__(self, DummyRewriter)
        return "dummy"


def test_base_query_rewriter_subclass():
    rw = DummyRewriter()
    assert rw.rewrite("abc") == ["ABC"]
    assert rw.model_name == "dummy"


class _FakeTensor:
    """Lightweight tensor-like helper for LLM fakes."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        if isinstance(self.data, list):
            return (len(self.data),)
        return ()

    def __getitem__(self, item):
        result = self.data[item]
        if isinstance(result, list):
            return _FakeTensor(result)
        return result

    def __iter__(self):
        for row in self.data:
            yield _FakeTensor(row) if isinstance(row, list) else row


class _FakeBatch(dict):
    """Minimal HuggingFace BatchEncoding stand-in."""

    def to(self, _device):
        return self


class _FakeTokenizer:
    """Tokenizer stub that records decode outputs."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.decode_outputs = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        user_msg = messages[-1]["content"]
        return f"{messages[0]['role']}:{user_msg}:{tokenize}:{add_generation_prompt}"

    def __call__(self, _prompt, **_kwargs):
        return _FakeBatch({"input_ids": _FakeTensor([[0, 1, 2, 3]])})

    def decode(self, _ids, skip_special_tokens=True):
        if not self.decode_outputs:
            return ""
        return self.decode_outputs.pop(0)


class _FakeCausalLM:
    """Causal LM stub with configurable outputs."""

    def __init__(self):
        self.generated_sequences = [_FakeTensor([0, 1, 2, 3, 4, 5])]

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **_kwargs):
        return self.generated_sequences


class _FakeSeq2SeqLM(_FakeCausalLM):
    """Seq2Seq stub that mimics returning multiple beams."""

    def generate(self, num_return_sequences=1, **_kwargs):
        return self.generated_sequences[:num_return_sequences]


@pytest.fixture
def fake_llm_runtime(monkeypatch):
    """Patch transformer classes with lightweight stubs for deterministic tests."""

    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.AutoTokenizer",
        _FakeTokenizer,
    )
    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.AutoModelForCausalLM",
        _FakeCausalLM,
    )
    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.AutoModelForSeq2SeqLM",
        _FakeSeq2SeqLM,
    )
    return {"tokenizer_cls": _FakeTokenizer, "causal_cls": _FakeCausalLM}


def test_llm_rewriter_parses_canonical_json(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"is_canonical": false, "rewritten_queries": ["paper one", "paper two"]}'
    ]
    result = rewriter.rewrite("original resnet paper", num_rewrites=2)

    assert result["canonical_intent"] is True  # pattern fallback flips False -> True
    assert result["rewrites"] == ["paper one", "paper two"]


def test_llm_rewriter_falls_back_to_pattern_detection(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["not json at all"]
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2, 3, 4, 5, 6])]

    result = rewriter.rewrite("original unet paper", num_rewrites=2)
    assert result["canonical_intent"] is True
    assert result["rewrites"][0] == "not json at all"
    assert result["rewrites"][1] == "original unet paper"


def test_extract_filters_and_rewrite_success(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        """```json
        {
          "filters": {"year": {"min": 2019, "max": 2023}, "citation_count": {"min": 500, "max": null}},
          "confidence": 1.5,
          "rewrites": ["state of the art vision transformer", "vit sota papers"]
        }
        ```"""
    ]

    result = rewriter.extract_filters_and_rewrite(
        "state of the art vision transformer", num_rewrites=1, current_year=2024
    )

    assert result["filters"]["year"] == {"min": 2019, "max": 2023}
    assert result["filters"]["citation_count"]["min"] == 500
    assert result["confidence"] == 1.0  # clamped
    assert result["rewrites"] == ["state of the art vision transformer"]


def test_extract_filters_and_rewrite_fallback(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["missing json braces"]

    result = rewriter.extract_filters_and_rewrite("graph neural nets", num_rewrites=2)

    assert result["filters"]["year"] == {"min": None, "max": None}
    assert result["confidence"] == 0.0
    assert result["rewrites"] == ["graph neural nets"]


def test_seq2seq_mode_returns_plain_rewrites(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="flan-t5-small", device="cpu", canonical_intent_enabled=False
    )
    rewriter.tokenizer.decode_outputs = ["expanded query"]
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2])]

    rewrites = rewriter.rewrite("bert nlp basics", num_rewrites=1)
    assert rewrites == ["expanded query"]


def test_llm_rewriter_defaults_to_cpu(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device=None)
    assert rewriter.device == "cpu"
    assert rewriter.model_name == "fake-model"


def test_detect_canonical_fallback_false(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    assert rewriter._detect_canonical_fallback("what is attention mechanism") is False


def test_build_prompt_without_chat_template(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="fake-model", device="cpu", canonical_intent_enabled=False
    )
    prompt = rewriter._build_prompt("graph neural nets", num_rewrites=2)
    assert "Rewrite this search query" in prompt


def test_build_prompt_canonical_no_chat_template(monkeypatch, fake_llm_runtime):
    class NoChatTokenizer(_FakeTokenizer):
        def __getattribute__(self, name):
            if name == "apply_chat_template":
                raise AttributeError
            return super().__getattribute__(name)

    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer = NoChatTokenizer()
    prompt = rewriter._build_prompt("canonical question", num_rewrites=1)
    assert prompt.endswith("Assistant:")


def test_rewrite_uses_default_num_rewrites(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu", num_rewrites=3)
    rewriter.tokenizer.decode_outputs = [
        '{"is_canonical": false, "rewritten_queries": ["a better query"]}'
    ]
    result = rewriter.rewrite("original", num_rewrites=None)
    assert result["rewrites"][0] == "a better query"
    assert len(result["rewrites"]) == 3


def test_rewrite_empty_generated_text_defaults_to_query(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["   "]
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2, 3, 4])]
    result = rewriter.rewrite("seminal paper")
    assert result["rewrites"][0] == "seminal paper"


def test_rewrite_causal_lm_without_canonical(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="fake-model",
        device="cpu",
        canonical_intent_enabled=False,
    )
    rewriter.tokenizer.decode_outputs = ["non canonical rewrite"]
    rewrites = rewriter.rewrite("search topic", num_rewrites=1)
    assert rewrites == ["non canonical rewrite"]


def test_extract_filters_without_chat_template(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"filters": {"year": {"min": 2010, "max": 2015}, "citation_count": {"min": 100, "max": null}}, "confidence": 0.7, "rewrites": ["foo"]}'
    ]
    result = rewriter.extract_filters_and_rewrite("foo query", num_rewrites=None)
    assert result["filters"]["year"]["min"] == 2010
    assert result["rewrites"] == ["foo"]


def test_extract_filters_invalid_structure(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"filters": {"year": 2020, "citation_count": "high"}, "confidence": 0.9, "rewrites": "none"}'
    ]
    result = rewriter.extract_filters_and_rewrite("fallback query")
    assert result["filters"]["year"] == {"min": None, "max": None}
    assert result["filters"]["citation_count"]["min"] is None
    assert result["rewrites"] == ["fallback query"]


def test_extract_filters_no_chat_template(monkeypatch, fake_llm_runtime):
    class NoChatTokenizer(_FakeTokenizer):
        def __getattribute__(self, name):
            if name == "apply_chat_template":
                raise AttributeError
            return super().__getattribute__(name)
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.AutoTokenizer",
        NoChatTokenizer,
    )
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"filters": {"year": {"min": 2012, "max": 2016}, "citation_count": {"min": 50, "max": null}}, "confidence": 0.6, "rewrites": ["alt"]}'
    ]
    result = rewriter.extract_filters_and_rewrite("multi query")
    assert result["filters"]["year"]["min"] == 2012
    assert result["rewrites"] == ["alt"]


def test_extract_filters_seq2seq_mode(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="flan-t5-small", device="cpu")
    result = rewriter.extract_filters_and_rewrite("seq2seq test")
    assert result["confidence"] == 0.0
    assert isinstance(result["rewrites"], list)
