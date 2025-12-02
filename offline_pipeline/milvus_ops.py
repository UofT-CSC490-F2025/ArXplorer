from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
from pymilvus import (  # type: ignore
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
)

logger = logging.getLogger(__name__)


DENSE_COLLECTION = "dense_collection"
SPARSE_COLLECTION = "sparse_collection"


def connect(host: str, port: str | int):
    logger.info("Connecting to Milvus %s:%s", host, port)
    connections.connect(alias="default", host=host, port=str(port))


def ensure_collections(dimension: int = 768):
    fields_dense = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="year", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="citation_count", dtype=DataType.INT64, is_nullable=True),
    ]
    fields_sparse = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]

    if DENSE_COLLECTION not in list_collections():
        logger.info("Creating dense collection")
        Collection(name=DENSE_COLLECTION, schema=CollectionSchema(fields=fields_dense))
    if SPARSE_COLLECTION not in list_collections():
        logger.info("Creating sparse collection")
        Collection(name=SPARSE_COLLECTION, schema=CollectionSchema(fields=fields_sparse))


def list_collections() -> list[str]:
    from pymilvus import utility  # type: ignore

    return utility.list_collections()


def create_indexes():
    dense = Collection(DENSE_COLLECTION)
    sparse = Collection(SPARSE_COLLECTION)
    dense.create_index(field_name="dense_vector", index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
    sparse.create_index(field_name="sparse_vector", index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}})
    dense.load()
    sparse.load()


def upsert_dense(records: Iterable[dict], embeddings: np.ndarray):
    dense = Collection(DENSE_COLLECTION)
    ids = [r["id"] for r in records]
    years = [r.get("year") or 0 for r in records]
    citations = [r.get("citation_count") or 0 for r in records]
    dense.insert([ids, embeddings, years, citations])


def upsert_sparse(records: List[dict], indices: List[List[int]], values: List[List[float]]):
    sparse = Collection(SPARSE_COLLECTION)
    ids = [r["id"] for r in records]
    try:
        sparse.insert([ids, {"indices": indices, "values": values}])
    except MilvusException as exc:  # noqa: BLE001
        logger.error("Sparse upsert failed: %s", exc)
        raise
