import numpy as np
import pytest

from src.data.document import Document
from src.retrieval.indexers.milvus_indexer import MilvusIndexer


class DummyDenseEncoder:
    def get_dimension(self):
        return 2

    def encode(self, texts, batch_size=None):
        return np.array([[len(text), len(text) + 1] for text in texts], dtype=np.float32)


class DummySparseEncoder:
    def encode(self, texts, batch_size=4):
        return [
            (np.array([0, len(text)], dtype=np.int64), np.array([1.0, float(len(text))], dtype=np.float32))
            for text in texts
        ]


class DummyCollection:
    def __init__(self, name, schema=None, using=None):
        self.name = name
        self.schema = schema
        self.inserted_batches = []
        self.indexes = []
        self.flushed = False
        self.loaded = False
        self.num_entities = 0
        self.deleted_exprs = []

    def insert(self, entities):
        self.inserted_batches.append(entities)
        self.num_entities += len(entities)

    def create_index(self, field_name, index_params):
        self.indexes.append((field_name, index_params))

    def flush(self):
        self.flushed = True

    def load(self):
        self.loaded = True

    def delete(self, expr):
        self.deleted_exprs.append(expr)


def setup_milvus_fixtures(monkeypatch, *, collection_exists=False):
    def fake_connect(alias, host, port):
        return None

    class DummyUtility:
        def __init__(self, exists):
            self.exists = exists
            self.dropped = []

        def has_collection(self, name):
            return self.exists

        def drop_collection(self, name):
            self.dropped.append(name)

    utility = DummyUtility(collection_exists)

    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.connections.connect", fake_connect)
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.utility", utility)
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.Collection", DummyCollection)
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.CollectionSchema", lambda fields, description, enable_dynamic_field: ("schema", fields))
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.FieldSchema", lambda **kwargs: kwargs)

    return utility


def test_add_documents_creates_collection_and_inserts(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    dense = DummyDenseEncoder()
    sparse = DummySparseEncoder()

    indexer = MilvusIndexer(
        dense_encoder=dense,
        sparse_encoder=sparse,
        output_dir="tmp",
        host="localhost",
        port=19530,
        collection_name="test_collection",
        batch_size=2,
        use_metadata=True,
        metadata_template="Title: {title} | Abstract: {abstract} | Authors: {authors} | Categories: {categories} | Year: {year}",
    )

    docs = [
        Document(id="1", text="doc one", title="Title1", metadata=None, published_year=2020),
        Document(id="2", text="doc two", title="Title2", metadata=None, published_year=2021),
    ]
    docs[0].authors = ["Alice"]
    docs[0].categories = ["cs.AI"]
    docs[1].authors = ["Bob"]
    docs[1].categories = ["cs.LG"]

    indexer.add_documents(docs, batch_size=1, show_progress=False)

    assert indexer.collection is not None
    total_inserted = sum(len(batch) for batch in indexer.collection.inserted_batches)
    assert total_inserted == 2
    first_entity = indexer.collection.inserted_batches[0][0]
    assert first_entity["id"] == "1"
    assert "doc one" in first_entity["abstract"]
    assert isinstance(first_entity["dense_vector"], list)
    assert len(first_entity["dense_vector"]) == 2
    assert len(first_entity["sparse_vector"]) == 2
    assert 0 in first_entity["sparse_vector"]
    other_keys = [k for k in first_entity["sparse_vector"].keys() if k != 0]
    assert other_keys and other_keys[0] > 0


def test_add_documents_handles_empty(monkeypatch, capsys):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    indexer.add_documents([], show_progress=False)
    out = capsys.readouterr().out
    assert "No documents to add" in out


def test_save_creates_indexes(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    docs = [Document(id="1", text="hello")]
    indexer.add_documents(docs, show_progress=False)

    indexer.save()
    assert indexer.collection.flushed
    assert indexer.collection.loaded
    # two indexes created: dense and sparse
    assert len(indexer.collection.indexes) == 2


def test_load_existing_collection(monkeypatch):
    def fake_has_collection(name):
        return True

    setup_milvus_fixtures(monkeypatch, collection_exists=True)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    indexer.load()
    assert indexer.collection is not None
    assert indexer.collection.loaded


def test_load_missing_collection_raises(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    with pytest.raises(ValueError):
        indexer.load()


def test_connect_failure(monkeypatch):
    def failing_connect(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.connections.connect", failing_connect)
    with pytest.raises(ConnectionError):
        MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")


def test_create_collection_drops_existing(monkeypatch):
    utility = setup_milvus_fixtures(monkeypatch, collection_exists=True)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    indexer._create_collection()
    assert utility.dropped == ["arxplorer_papers"]


def test_create_indexes_variants(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    hnsw_indexer = MilvusIndexer(
        DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp", dense_index_type="HNSW"
    )
    hnsw_indexer._create_collection()
    hnsw_indexer._create_indexes()
    assert any(idx[1]["index_type"] == "HNSW" for idx in hnsw_indexer.collection.indexes)

    flat_indexer = MilvusIndexer(
        DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp", dense_index_type="FLAT"
    )
    flat_indexer._create_collection()
    flat_indexer._create_indexes()
    assert any(idx[1]["index_type"] == "FLAT" for idx in flat_indexer.collection.indexes)


def test_add_documents_with_progress(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.tqdm", lambda iterable, **kwargs: iterable)

    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp", batch_size=1)
    docs = [Document(id="a", text="aaa"), Document(id="b", text="bbbb")]
    indexer.add_documents(docs, batch_size=1, show_progress=True)
    assert indexer.collection.num_entities == 2


def test_convert_sparse_to_milvus():
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    converted = indexer._convert_sparse_to_milvus([
        (np.array([0, 2]), np.array([1.0, 3.0]))
    ])
    assert converted == [{0: 1.0, 2: 3.0}]


def test_save_without_collection(monkeypatch, capsys):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    indexer.collection = None
    indexer.save()
    assert "No collection to save" in capsys.readouterr().out


def test_get_num_documents(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    assert indexer.get_num_documents() == 0
    indexer._create_collection()
    indexer.collection.num_entities = 5
    assert indexer.get_num_documents() == 5


def test_update_citation_counts(monkeypatch):
    setup_milvus_fixtures(monkeypatch, collection_exists=False)
    monkeypatch.setattr("src.retrieval.indexers.milvus_indexer.tqdm", lambda iterable, **kwargs: iterable)
    indexer = MilvusIndexer(DummyDenseEncoder(), DummySparseEncoder(), output_dir="tmp")
    indexer._create_collection()
    indexer.update_citation_counts({"doc1": 10, "doc2": 5})
    assert len(indexer.collection.deleted_exprs) == 2
