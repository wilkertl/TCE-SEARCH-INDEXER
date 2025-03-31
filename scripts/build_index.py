#!/usr/bin/env python3
"""
Script to build an index from documents
"""
import os
import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path so we can import the legal_indexer package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_indexer.chunker import DocumentChunker
from legal_indexer.embedder import DocumentEmbedder
from legal_indexer.indexer import DocumentIndex, LegalDocumentIndexer
from legal_indexer.data_loaders import load_documents
from legal_indexer.utils import (
    load_model_and_tokenizer,
    get_embedding_dim,
    create_directory_if_not_exists,
    format_time
)


def main():
    parser = argparse.ArgumentParser(description="Build a document index")

    # Required parameters
    parser.add_argument(
        "--documents_path",
        type=str,
        required=True,
        help="Path to document file or directory"
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory to save the index"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="BAAI/bge-m3",
        help="Path to the model (default: BAAI/bge-m3)"
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=1024,
        help="Embedding dimension (default: 1024)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (default: auto-detect)"
    )

    # Chunking parameters
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Maximum token length for each chunk (default: 512)"
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Number of tokens to overlap between chunks (default: 50)"
    )

    # Index parameters
    parser.add_argument(
        "--index_type",
        type=str,
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="Type of FAISS index to use (default: flat)"
    )

    # Document loading parameters
    parser.add_argument(
        "--id_field",
        type=str,
        default="id",
        help="Field name for document ID in JSON/CSV files (default: id)"
    )

    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field name for document text in JSON/CSV files (default: text)"
    )

    # Performance parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for encoding (default: 8)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress information"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create index directory if it doesn't exist
    create_directory_if_not_exists(args.index_dir)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)

    # Create components
    chunker = DocumentChunker(
        tokenizer=tokenizer,
        max_chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    embedder = DocumentEmbedder(
        model=model,
        tokenizer=tokenizer,
        max_length=args.chunk_size,
        device=args.device,
        batch_size=args.batch_size
    )

    # Check if index exists
    if os.path.exists(os.path.join(args.index_dir, "index.faiss")):
        print(f"Loading existing index from {args.index_dir}")

        # Load the index
        index = DocumentIndex.load(args.index_dir)

        # Create the indexer with existing index
        indexer = LegalDocumentIndexer(chunker=chunker, embedder=embedder, index=index)
    else:
        # Create a new index
        index = DocumentIndex(
            embedding_dim=args.embedding_dim,
            index_type=args.index_type
        )

        # Create the indexer with new index
        indexer = LegalDocumentIndexer(chunker=chunker, embedder=embedder, index=index)

    # Load documents
    start_time = time.time()
    documents = load_documents(
        args.documents_path,
        id_field=args.id_field,
        text_field=args.text_field
    )
    print(f"Loaded {len(documents)} documents in {format_time(time.time() - start_time)}")

    if not documents:
        print("No documents loaded. Check file format and field names.")
        return

    # Process documents
    start_time = time.time()
    indexer.process_documents(
        documents=documents,
        batch_size=args.batch_size,
        index_dir=args.index_dir,
        show_progress=not args.quiet
    )
    print(f"Indexed documents in {format_time(time.time() - start_time)}")

    # Save the index
    indexer.save(args.index_dir)
    print(f"Index saved to {args.index_dir}")


if __name__ == "__main__":
    main()