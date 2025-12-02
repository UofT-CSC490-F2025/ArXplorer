from __future__ import annotations

import logging
import sys
import traceback
from typing import List

from .config import PipelineConfig
from .embedding import dense_embed, load_dense_model, load_sparse_model, sparse_embed
from .milvus_ops import (
    connect as milvus_connect,
    create_indexes,
    ensure_collections,
    upsert_dense,
    upsert_sparse,
)
from .openalex_client import enrich_with_openalex
from .preprocess import build_text_template, filter_and_normalize
from .s3_io import list_keys, s3_client, stream_jsonl_from_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run():
    logger.info("Offline pipeline starting")
    cfg = PipelineConfig.from_env()
    client = s3_client(cfg.aws_region)

    # 1) Read raw JSONL(s) from S3
    raw_keys = list_keys(client, cfg.raw_bucket, "")
    if not raw_keys:
        raise SystemExit(f"No objects found in s3://{cfg.raw_bucket}/")

    logger.info("Found %d raw files; starting stream", len(raw_keys))
    raw_records_iter = (rec for key in raw_keys for rec in stream_jsonl_from_s3(client, cfg.raw_bucket, key))

    # 2) Filter + normalize + optional limit
    filtered = list(filter_and_normalize(raw_records_iter, cfg.max_samples))
    logger.info("Filtered to %d ML records", len(filtered))

    # 3) OpenAlex enrich
    enriched = list(enrich_with_openalex(filtered, email=cfg.openalex_email, rate_limit=cfg.openalex_rate_limit))

    # 4) Build text templates
    texts = [build_text_template(r) for r in enriched]

    # 5) Embeddings
    dense_model = load_dense_model(cfg.dense_model)
    dense_vecs = dense_embed(dense_model, texts, cfg.batch_size)

    tokenizer, sparse_model = load_sparse_model(cfg.sparse_model)
    sparse_indices, sparse_values = sparse_embed(tokenizer, sparse_model, texts, cfg.batch_size)

    # 6) Milvus upsert
    milvus_connect(cfg.milvus_host, cfg.milvus_port)
    ensure_collections(dimension=dense_vecs.shape[1])
    upsert_dense(enriched, dense_vecs)
    upsert_sparse(enriched, sparse_indices, sparse_values)
    create_indexes()

    logger.info("Pipeline completed: %d records processed", len(enriched))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        sys.exit(1)
