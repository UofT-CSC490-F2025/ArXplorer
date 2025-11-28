from pathlib import Path

import yaml

from config import Config, EncoderConfig, IndexConfig, SearchConfig, RerankerConfig, DataConfig


BASE_TMP = Path(__file__).parent / "tmp_config"


def make_tmp_file(name: str) -> Path:
    BASE_TMP.mkdir(exist_ok=True)
    return BASE_TMP / name


def test_default_config_has_expected_defaults():
    cfg = Config.default()

    assert isinstance(cfg.encoder, EncoderConfig)
    assert isinstance(cfg.index, IndexConfig)
    assert isinstance(cfg.search, SearchConfig)
    assert isinstance(cfg.reranker, RerankerConfig)
    assert isinstance(cfg.data, DataConfig)

    # Spot-check a few key defaults
    assert cfg.encoder.dense_model == "sentence-transformers/allenai-specter"
    assert cfg.encoder.sparse_model == "naver/splade-v3"
    assert cfg.search.top_k == 10
    assert cfg.data.jsonl_file == "data/arxiv_1k.jsonl"
    assert cfg.milvus.collection_name == "arxplorer_papers"


def test_from_yaml_overrides_nested_fields():
    yaml_path = make_tmp_file("override_config.yaml")
    yaml_content = {
        "encoder": {
            "dense_model": "my-dense-model",
            "sparse_model": "my-sparse-model",
            "use_specter2": True,
        },
        "index": {
            "batch_size": 32,
            "dense_output_dir": "out/dense",
        },
        "search": {
            "top_k": 25,
            "retrieval_k": 200,
        },
        "reranker": {
            "enabled": False,
            "model": "my-reranker-model",
        },
        "citation": {
            "enabled": True,
            "data_file": "data/citations_custom.json",
        },
        "query_rewriting": {
            "enabled": True,
            "model": "my-llm",
            "num_rewrites": 3,
        },
        "data": {
            "jsonl_file": "data/custom.jsonl",
            "text_key": "body",
            "id_key": "paper_id",
        },
        "milvus": {
            "host": "milvus-host",
            "port": 20000,
            "collection_name": "custom_collection",
        },
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content), encoding="utf-8")

    cfg = Config.from_yaml(str(yaml_path))

    assert cfg.encoder.dense_model == "my-dense-model"
    assert cfg.encoder.sparse_model == "my-sparse-model"
    assert cfg.encoder.use_specter2 is True

    assert cfg.index.batch_size == 32
    assert cfg.index.dense_output_dir == "out/dense"

    assert cfg.search.top_k == 25
    assert cfg.search.retrieval_k == 200

    assert cfg.reranker.enabled is False
    assert cfg.reranker.model == "my-reranker-model"

    assert cfg.citation.enabled is True
    assert cfg.citation.data_file == "data/citations_custom.json"

    assert cfg.query_rewriting.enabled is True
    assert cfg.query_rewriting.model == "my-llm"
    assert cfg.query_rewriting.num_rewrites == 3

    assert cfg.data.jsonl_file == "data/custom.jsonl"
    assert cfg.data.text_key == "body"
    assert cfg.data.id_key == "paper_id"

    assert cfg.milvus.host == "milvus-host"
    assert cfg.milvus.port == 20000
    assert cfg.milvus.collection_name == "custom_collection"


def test_from_yaml_missing_file_raises():
    missing = make_tmp_file("does_not_exist.yaml")
    if missing.exists():
        missing.unlink()

    try:
        Config.from_yaml(str(missing))
    except FileNotFoundError as e:
        assert "Config file not found" in str(e)
    else:
        raise AssertionError("Expected FileNotFoundError for missing config file")


def test_from_yaml_empty_file_uses_defaults():
    yaml_path = make_tmp_file("empty_config.yaml")
    yaml_path.write_text("", encoding="utf-8")

    cfg = Config.from_yaml(str(yaml_path))
    default_cfg = Config.default()

    # Spot-check that empty YAML behaves like default for a few fields
    assert cfg.encoder.dense_model == default_cfg.encoder.dense_model
    assert cfg.search.top_k == default_cfg.search.top_k
    assert cfg.data.jsonl_file == default_cfg.data.jsonl_file
    assert cfg.milvus.collection_name == default_cfg.milvus.collection_name


def test_to_yaml_writes_expected_structure():
    cfg = Config.default()
    out_path = make_tmp_file("roundtrip.yaml")

    if out_path.exists():
        out_path.unlink()

    cfg.to_yaml(str(out_path))
    assert out_path.exists()

    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    # Top-level sections
    assert set(data.keys()) == {"encoder", "index", "search", "reranker", "data"}

    # Encoder section matches selected fields
    enc = data["encoder"]
    assert enc["dense_model"] == cfg.encoder.dense_model
    assert enc["sparse_model"] == cfg.encoder.sparse_model
    assert enc["max_length"] == cfg.encoder.max_length
    assert enc["normalize_dense"] is cfg.encoder.normalize_dense
    assert enc["device"] == cfg.encoder.device

    # Index section
    idx = data["index"]
    assert idx["batch_size"] == cfg.index.batch_size
    assert idx["dense_output_dir"] == cfg.index.dense_output_dir
    assert idx["sparse_output_dir"] == cfg.index.sparse_output_dir
    assert idx["use_gpu_faiss"] is cfg.index.use_gpu_faiss
    assert idx["checkpoint_enabled"] is cfg.index.checkpoint_enabled

    # Search, reranker, data sections exist and at least one field matches
    assert data["search"]["top_k"] == cfg.search.top_k
    assert data["reranker"]["enabled"] == cfg.reranker.enabled
    assert data["data"]["jsonl_file"] == cfg.data.jsonl_file
