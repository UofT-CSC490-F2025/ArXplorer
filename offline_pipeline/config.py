from dataclasses import dataclass
import os


def _get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise ValueError(f"Missing required env var: {name}")
    return val


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass
class PipelineConfig:
    aws_region: str
    raw_bucket: str
    processed_bucket: str
    metadata_bucket: str
    embeddings_bucket: str
    milvus_host: str
    milvus_port: int
    dense_model: str = "allenai/specter2_base"
    sparse_model: str = "naver/splade-cocondenser-ensembledistil"
    batch_size: int = 8
    max_samples: int | None = None
    openalex_email: str | None = None
    openalex_rate_limit: float = 2.0  # req/sec

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            aws_region=_get_env("AWS_REGION", "ca-central-1"),
            raw_bucket=_get_env("RAW_BUCKET"),
            processed_bucket=_get_env("PROCESSED_BUCKET"),
            metadata_bucket=_get_env("METADATA_BUCKET"),
            embeddings_bucket=_get_env("EMBEDDINGS_BUCKET"),
            milvus_host=_get_env("MILVUS_HOST"),
            milvus_port=int(_get_env("MILVUS_PORT", "19530")),
            dense_model=os.getenv("DENSE_MODEL", "allenai/specter2_base"),
            sparse_model=os.getenv("SPARSE_MODEL", "naver/splade-cocondenser-ensembledistil"),
            batch_size=_get_int("BATCH_SIZE", 8),
            max_samples=int(os.getenv("MAX_SAMPLES")) if os.getenv("MAX_SAMPLES") else None,
            openalex_email=os.getenv("OPENALEX_EMAIL"),
            openalex_rate_limit=float(os.getenv("OPENALEX_RATE_LIMIT", "2.0")),
        )
