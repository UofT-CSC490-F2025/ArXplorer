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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
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
    return {"tokenizer_cls": _FakeTokenizer, "causal_cls": _FakeCausalLM}


def test_llm_rewriter_parses_canonical_json(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "topical", "rewrites": ["paper one", "paper two"]}'
    ]
    result = rewriter.rewrite("original resnet paper", num_rewrites=2)

    # rewrite() returns a list, not a dict
    assert isinstance(result, list)
    assert result == ["paper one", "paper two"]


def test_llm_rewriter_falls_back_to_pattern_detection(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["not json at all"]
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2, 3, 4, 5, 6])]

    result = rewriter.rewrite("original unet paper", num_rewrites=2)
    # rewrite() returns a list
    assert isinstance(result, list)
    assert len(result) == 2


def test_extract_intent_filters_and_rewrite_success(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        """```json
        {
          "intent": "sota",
          "year_constraint": {"min": 2019, "max": 2023},
          "citation_threshold": 500,
          "rewrites": ["state of the art vision transformer"]
        }
        ```"""
    ]

    result = rewriter.extract_intent_filters_and_rewrite(
        "state of the art vision transformer", num_rewrites=1, current_year=2024
    )

    assert result["year_constraint"] == {"min": 2019, "max": 2023}
    assert result["citation_threshold"] == 500
    assert result["intent"] == "sota"
    assert result["rewrites"] == ["state of the art vision transformer"]


def test_extract_intent_filters_and_rewrite_fallback(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["missing json braces"]

    result = rewriter.extract_intent_filters_and_rewrite("graph neural nets", num_rewrites=2)

    # Fallback returns dict with default values
    assert result["year_constraint"] == {"min": None, "max": None}
    assert result["intent"] == "topical"
    assert "graph neural nets" in result["rewrites"]


def test_seq2seq_mode_returns_plain_rewrites(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="flan-t5-small", device="cpu"
    )
    rewriter.tokenizer.decode_outputs = ['{"intent": "topical", "rewrites": ["expanded query"]}']
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2])]

    rewrites = rewriter.rewrite("bert nlp basics", num_rewrites=1)
    assert rewrites == ["expanded query"]


def test_llm_rewriter_defaults_to_cpu(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device=None)
    assert rewriter.device == "cpu"
    assert rewriter.model_name == "fake-model"


@pytest.mark.skip(reason="_detect_canonical_fallback method no longer exists in current implementation")
def test_detect_canonical_fallback_false(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    assert rewriter._detect_canonical_fallback("what is attention mechanism") is False


@pytest.mark.skip(reason="_build_prompt method no longer exists in current implementation")
def test_build_prompt_without_chat_template(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="fake-model", device="cpu", canonical_intent_enabled=False
    )
    prompt = rewriter._build_prompt("graph neural nets", num_rewrites=2)
    assert "Rewrite this search query" in prompt


@pytest.mark.skip(reason="_build_prompt method no longer exists in current implementation")
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
        '{"intent": "topical", "rewrites": ["a better query"]}'
    ]
    result = rewriter.rewrite("original", num_rewrites=None)
    # rewrite() returns a list
    assert isinstance(result, list)
    assert "a better query" in result
    assert len(result) == 3


def test_rewrite_empty_generated_text_defaults_to_query(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["   "]
    rewriter.model.generated_sequences = [_FakeTensor([0, 1, 2, 3, 4])]
    result = rewriter.rewrite("seminal paper")
    # rewrite() returns a list, fallback should include original query
    assert isinstance(result, list)
    assert "seminal paper" in result


def test_rewrite_causal_lm_without_canonical(fake_llm_runtime):
    rewriter = LLMQueryRewriter(
        model_name="fake-model",
        device="cpu"
    )
    rewriter.tokenizer.decode_outputs = ['{"intent": "topical", "rewrites": ["non canonical rewrite"]}']
    rewrites = rewriter.rewrite("search topic", num_rewrites=1)
    assert rewrites == ["non canonical rewrite"]


def test_extract_filters_without_chat_template(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "topical", "year_constraint": {"min": 2010, "max": 2015}, "citation_threshold": 100, "rewrites": ["foo"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("foo query", num_rewrites=None)
    assert result["year_constraint"]["min"] == 2010
    assert result["rewrites"] == ["foo"]


def test_extract_filters_invalid_structure(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"year_constraint": 2020, "citation_threshold": "high", "rewrites": "none"}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("fallback query")
    # Invalid structure should fallback
    assert result["year_constraint"] == {"min": None, "max": None}
    assert result["citation_threshold"] is None
    assert "fallback query" in result["rewrites"]


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
        '{"intent": "topical", "year_constraint": {"min": 2012, "max": 2016}, "citation_threshold": 50, "rewrites": ["alt"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("multi query")
    assert result["year_constraint"]["min"] == 2012
    assert result["rewrites"] == ["alt"]


def test_extract_filters_seq2seq_mode(fake_llm_runtime):
    rewriter = LLMQueryRewriter(model_name="flan-t5-small", device="cpu")
    result = rewriter.extract_intent_filters_and_rewrite("seq2seq test")
    assert result["intent"] == "topical"  # fallback intent
    assert isinstance(result["rewrites"], list)


def test_parse_json_with_nested_braces(fake_llm_runtime):
    """Test parsing JSON that contains nested structures."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "topical", "year_constraint": {"min": 2020, "max": 2025}, "citation_threshold": 100, "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test query")
    assert result["year_constraint"]["min"] == 2020
    assert result["year_constraint"]["max"] == 2025


def test_parse_json_with_raw_decode_fallback(fake_llm_runtime):
    """Test JSON parsing that requires raw_decode fallback."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    # Simulate malformed JSON that needs raw_decode
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "sota", "rewrites": ["query"]} extra text'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["intent"] == "sota"


def test_parse_json_no_braces_regex_search(fake_llm_runtime):
    """Test JSON extraction when response doesn't start with brace."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        'Here is the result: {"intent": "comparison", "rewrites": ["q1", "q2"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test", num_rewrites=2)
    assert result["intent"] == "comparison"
    assert len(result["rewrites"]) == 2


def test_parse_json_no_json_found_raises_error(fake_llm_runtime):
    """Test parsing when no JSON is found in response."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        "No JSON here at all, just plain text"
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test query")
    # Should fallback
    assert result["intent"] in ["topical", "sota", "foundational", "comparison", "method_lookup", "specific_paper"]


def test_parse_target_title_non_string(fake_llm_runtime):
    """Test target_title handling when it's not a string."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_title": 123, "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_title"] is None


def test_parse_target_title_empty_string(fake_llm_runtime):
    """Test target_title handling when it's an empty string."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_title": "", "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_title"] is None


def test_parse_target_title_null_string(fake_llm_runtime):
    """Test target_title handling when it's the string 'null'."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_title": "null", "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_title"] is None


def test_parse_target_authors_non_list(fake_llm_runtime):
    """Test target_authors handling when it's not a list."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_authors": "not a list", "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_authors"] is None


def test_parse_target_authors_with_non_strings(fake_llm_runtime):
    """Test target_authors filtering of non-string elements."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_authors": ["Smith", 123, ""], "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_authors"] == ["Smith"]


def test_parse_target_authors_all_empty(fake_llm_runtime):
    """Test target_authors when all elements are empty after filtering."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "specific_paper", "target_authors": ["", "  ", 123], "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["target_authors"] is None


def test_parse_negative_citation_threshold(fake_llm_runtime):
    """Test citation_threshold handling when negative."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "sota", "citation_threshold": -50, "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["citation_threshold"] is None


def test_parse_invalid_citation_threshold_string(fake_llm_runtime):
    """Test citation_threshold when it's an invalid string."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "sota", "citation_threshold": "not a number", "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["citation_threshold"] is None


def test_fallback_detects_foundational_keywords(fake_llm_runtime):
    """Test fallback pattern detection for foundational intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("original transformer paper")
    assert result["intent"] == "foundational"


def test_fallback_detects_sota_keywords(fake_llm_runtime):
    """Test fallback pattern detection for SOTA intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("recent advances in NLP")
    assert result["intent"] == "sota"


def test_fallback_detects_comparison_keywords(fake_llm_runtime):
    """Test fallback pattern detection for comparison intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("bert vs gpt comparison")
    assert result["intent"] == "comparison"


def test_fallback_detects_method_lookup_keywords(fake_llm_runtime):
    """Test fallback pattern detection for method_lookup intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("what is attention mechanism")
    assert result["intent"] == "method_lookup"


def test_fallback_detects_specific_paper_keywords(fake_llm_runtime):
    """Test fallback pattern detection for specific_paper intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("resnet paper")
    assert result["intent"] == "specific_paper"


def test_fallback_defaults_to_topical(fake_llm_runtime):
    """Test fallback defaults to topical intent."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = ["invalid json"]
    result = rewriter.extract_intent_filters_and_rewrite("neural networks")
    assert result["intent"] == "topical"


def test_model_name_property(fake_llm_runtime):
    """Test model_name property returns correct value."""
    rewriter = LLMQueryRewriter(model_name="test-model-123", device="cpu")
    assert rewriter.model_name == "test-model-123"


def test_vllm_initialization_without_openai(monkeypatch, fake_llm_runtime):
    """Test vLLM initialization fails when OpenAI library not available."""
    # Simulate OpenAI not available
    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.OPENAI_AVAILABLE", False
    )
    with pytest.raises(ImportError, match="OpenAI library required"):
        LLMQueryRewriter(model_name="fake-model", device="cpu", use_vllm=True)


def test_bedrock_initialization_without_boto3(monkeypatch, fake_llm_runtime):
    """Test Bedrock initialization fails when boto3 not available."""
    # Simulate boto3 not available
    monkeypatch.setattr(
        "src.retrieval.query_rewriting.llm_rewriter.BOTO3_AVAILABLE", False
    )
    with pytest.raises(ImportError, match="boto3 library required"):
        LLMQueryRewriter(model_name="fake-model", device="cpu", use_bedrock=True)


def test_parse_year_constraint_non_dict(fake_llm_runtime):
    """Test year_constraint handling when it's not a dict."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "topical", "year_constraint": "not a dict", "rewrites": ["test"]}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("test")
    assert result["year_constraint"] == {"min": None, "max": None}


def test_parse_rewrites_not_list(fake_llm_runtime):
    """Test rewrites handling when it's not a list."""
    rewriter = LLMQueryRewriter(model_name="fake-model", device="cpu")
    rewriter.tokenizer.decode_outputs = [
        '{"intent": "topical", "rewrites": "not a list"}'
    ]
    result = rewriter.extract_intent_filters_and_rewrite("fallback query", num_rewrites=2)
    # Should use fallback query
    assert "fallback query" in result["rewrites"]
    assert len(result["rewrites"]) == 2
