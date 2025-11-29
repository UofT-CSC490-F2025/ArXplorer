"""Migration script to load existing data into Milvus."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data import StreamingJSONLLoader, Document
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.indexers.milvus_indexer import MilvusIndexer


def migrate_to_milvus(config_path: str = "config.yaml", data_file: str = None):
    """
    Load existing JSONL data into Milvus collection.
    
    Args:
        config_path: Path to config.yaml
        data_file: Override JSONL data file path
    """
    print("="*70)
    print("ArXplorer: Migrate to Milvus")
    print("="*70)
    
    # Load config
    config = Config.from_yaml(config_path)
    
    if data_file:
        config.data.jsonl_file = data_file
    
    print(f"\nConfiguration:")
    print(f"  Data file: {config.data.jsonl_file}")
    print(f"  Dense model: {config.encoder.dense_model}")
    print(f"  Sparse model: {config.encoder.sparse_model}")
    print(f"  Milvus host: {config.milvus.host}:{config.milvus.port}")
    print(f"  Collection: {config.milvus.collection_name}")
    
    # Initialize encoders
    print("\nInitializing encoders...")
    dense_encoder = DenseEncoder(
        model_name=config.encoder.dense_model,
        device=config.encoder.device,
        normalize=config.encoder.normalize_dense,
        use_specter2=config.encoder.use_specter2,
        specter2_base_adapter=config.encoder.specter2_base_adapter,
        specter2_query_adapter=config.encoder.specter2_query_adapter
    )
    
    sparse_encoder = SparseEncoder(
        model_name=config.encoder.sparse_model,
        device=config.encoder.device,
        max_length=config.encoder.max_length
    )
    
    # Initialize Milvus indexer
    print("\nConnecting to Milvus...")
    indexer = MilvusIndexer(
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        output_dir=str(config.index.dense_output_dir),  # For checkpoints only
        host=config.milvus.host,
        port=config.milvus.port,
        collection_name=config.milvus.collection_name,
        dense_index_type=config.milvus.dense_index_type,
        dense_nlist=config.milvus.dense_nlist,
        batch_size=config.milvus.batch_size,
        use_metadata=config.data.use_metadata,
        metadata_template=config.data.metadata_template
    )
    
    # Load documents
    print(f"\nLoading documents from {config.data.jsonl_file}...")
    loader = StreamingJSONLLoader(
        filepath=config.data.jsonl_file,
        text_key=config.data.text_key,
        id_key=config.data.id_key,
        title_key=config.data.title_key
    )
    
    # Stream documents in batches
    batch_size = 1000
    batch = []
    total_docs = 0
    
    for doc in loader.load():  # Call .load() to get iterator
        batch.append(doc)
        
        if len(batch) >= batch_size:
            indexer.add_documents(
                batch,
                batch_size=config.index.batch_size,
                show_progress=True
            )
            total_docs += len(batch)
            print(f"  Progress: {total_docs} documents indexed")
            batch = []
    
    # Add remaining documents
    if batch:
        indexer.add_documents(
            batch,
            batch_size=config.index.batch_size,
            show_progress=True
        )
        total_docs += len(batch)
    
    print(f"\n✓ Loaded {total_docs} documents")
    
    # Save (build indexes and load collection)
    print("\nBuilding indexes...")
    indexer.save()
    
    print("\n" + "="*70)
    print("Migration Complete!")
    print("="*70)
    print(f"  Collection: {config.milvus.collection_name}")
    print(f"  Documents: {indexer.get_num_documents()}")
    print(f"  Dense vectors: {dense_encoder.get_dimension()}D")
    print(f"  Sparse vectors: {sparse_encoder.get_dimension()}D")
    print("\nYou can now query using:")
    print(f"  python scripts/query.py --method milvus")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing data to Milvus"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        help="Override data file path (default: from config)"
    )
    
    args = parser.parse_args()
    
    migrate_to_milvus(
        config_path=args.config,
        data_file=args.data_file
    )


if __name__ == "__main__":
    main()
