"""Unified encoding script for building dense and/or sparse indexes.

Usage:
    # Encode with dense only
    python scripts/encode.py --method dense
    
    # Encode with sparse only
    python scripts/encode.py --method sparse
    
    # Encode with both (for hybrid search)
    python scripts/encode.py --method both
    
    # Use custom config
    python scripts/encode.py --config my_config.yaml --method both
    
    # Override specific settings
    python scripts/encode.py --data-file data/arxiv_300k.jsonl --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data import StreamingJSONLLoader
from src.retrieval.encoders import DenseEncoder, SparseEncoder
from src.retrieval.indexers import DenseIndexer, SparseIndexer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build dense and/or sparse indexes from JSONL corpus"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["dense", "sparse", "both"],
        help="Encoding method: dense, sparse, or both"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    
    # Data overrides
    parser.add_argument(
        "--data-file",
        type=str,
        help="Override input JSONL file path"
    )
    
    parser.add_argument(
        "--text-key",
        type=str,
        help="Override JSON key for document text"
    )
    
    parser.add_argument(
        "--id-key",
        type=str,
        help="Override JSON key for document ID"
    )
    
    # Index overrides
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size for encoding"
    )
    
    parser.add_argument(
        "--dense-output",
        type=str,
        help="Override dense index output directory"
    )
    
    parser.add_argument(
        "--sparse-output",
        type=str,
        help="Override sparse index output directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Override device (cuda or cpu)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override chunk size for sparse indexing"
    )
    
    parser.add_argument(
        "--sparse-encoder-batch-size",
        type=int,
        help="Override SPLADE encoder batch size (for VRAM management)"
    )
    
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Enable metadata enrichment (title + authors + categories + abstract)"
    )
    
    return parser.parse_args()


def encode_dense(config: Config, loader: StreamingJSONLLoader):
    """Encode documents with dense encoder."""
    print("\n" + "="*60)
    print("DENSE ENCODING")
    print("="*60)
    
    # Create encoder
    encoder = DenseEncoder(
        model_name=config.encoder.dense_model,
        device=config.encoder.device,
        normalize=config.encoder.normalize_dense,
        use_specter2=config.encoder.use_specter2,
        specter2_adapter=config.encoder.specter2_adapter
    )
    
    # Create indexer
    indexer = DenseIndexer(
        encoder=encoder,
        output_dir=config.index.dense_output_dir,
        use_gpu=config.index.use_gpu_faiss
    )
    
    # Load documents and encode
    print(f"\nLoading documents from: {config.data.jsonl_file}")
    documents = list(loader.load())
    print(f"Loaded {len(documents)} documents")
    
    print(f"\nEncoding with batch_size={config.index.batch_size}...")
    indexer.add_documents(
        documents=documents,
        batch_size=config.index.batch_size,
        show_progress=True
    )
    
    print("\nSaving dense index...")
    indexer.save()
    
    print(f"\n✓ Dense index saved to: {config.index.dense_output_dir}")


def encode_sparse(config: Config, loader: StreamingJSONLLoader):
    """Encode documents with sparse encoder."""
    print("\n" + "="*60)
    print("SPARSE ENCODING")
    print("="*60)
    
    # Create encoder
    encoder = SparseEncoder(
        model_name=config.encoder.sparse_model,
        device=config.encoder.device,
        max_length=config.encoder.max_length
    )
    
    # Create indexer
    indexer = SparseIndexer(
        encoder=encoder,
        output_dir=config.index.sparse_output_dir,
        chunk_size=config.index.chunk_size,
        sparse_encoder_batch_size=config.index.sparse_encoder_batch_size
    )
    
    # Load documents and encode
    print(f"\nLoading documents from: {config.data.jsonl_file}")
    documents = list(loader.load())
    print(f"Loaded {len(documents)} documents")
    
    print(f"\nEncoding with batch_size={config.index.batch_size}...")
    indexer.add_documents(
        documents=documents,
        batch_size=config.index.batch_size,
        show_progress=True
    )
    
    print("\nSaving sparse index...")
    indexer.save()
    
    print(f"\n✓ Sparse index saved to: {config.index.sparse_output_dir}")


def main():
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = Config.default()
    
    # Apply CLI overrides
    if args.data_file:
        config.data.jsonl_file = args.data_file
    if args.text_key:
        config.data.text_key = args.text_key
    if args.id_key:
        config.data.id_key = args.id_key
    if args.batch_size:
        config.index.batch_size = args.batch_size
    if args.dense_output:
        config.index.dense_output_dir = args.dense_output
    if args.sparse_output:
        config.index.sparse_output_dir = args.sparse_output
    if args.device:
        config.encoder.device = args.device
    if args.chunk_size:
        config.index.chunk_size = args.chunk_size
    if args.sparse_encoder_batch_size:
        config.index.sparse_encoder_batch_size = args.sparse_encoder_batch_size
    if args.use_metadata:
        config.data.use_metadata = True
    
    # Create data loader
    loader = StreamingJSONLLoader(
        filepath=config.data.jsonl_file,
        text_key=config.data.text_key,
        id_key=config.data.id_key,
        title_key=config.data.title_key,
        use_metadata=config.data.use_metadata,
        categories_key=config.data.categories_key,
        authors_key=config.data.authors_key,
        metadata_template=config.data.metadata_template
    )
    
    # Encode based on method
    if args.method == "dense":
        encode_dense(config, loader)
    elif args.method == "sparse":
        encode_sparse(config, loader)
    elif args.method == "both":
        encode_dense(config, loader)
        # Reload loader for sparse (iterator exhausted)
        loader = StreamingJSONLLoader(
            filepath=config.data.jsonl_file,
            text_key=config.data.text_key,
            id_key=config.data.id_key,
            title_key=config.data.title_key,
            use_metadata=config.data.use_metadata,
            categories_key=config.data.categories_key,
            authors_key=config.data.authors_key,
            metadata_template=config.data.metadata_template
        )
        encode_sparse(config, loader)
    
    print("\n" + "="*60)
    print("ENCODING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
