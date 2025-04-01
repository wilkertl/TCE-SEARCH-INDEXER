#!/usr/bin/env python3
"""
Script to build an index from documents
"""
import os
import argparse
import time
import sys
import requests
from pathlib import Path
from urllib.parse import urlparse

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


def download_dataset(url, output_path):
    """
    Download a dataset from a URL and save it to the specified path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get filename from URL if not provided
        if os.path.isdir(output_path):
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                # Default to CSV since bills dataset is in CSV format
                filename = "downloaded_dataset.csv"
            output_path = os.path.join(output_path, filename)

        # Write content to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded bills dataset (CSV) to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Build a document index")

    # Document source parameters (optional with defaults)
    document_group = parser.add_mutually_exclusive_group(required=False)
    document_group.add_argument(
        "--documents_path",
        type=str,
        default="./data/documents",
        help="Path to document file or directory (default: ./data/documents)"
    )
    document_group.add_argument(
        "--bills_dataset_url",
        type=str,
        default="https://drive.google.com/uc?id=1yuRCNKLphpk6FPhAfBD4_yygnOPFilHj",
        help="URL to download the bills dataset CSV"
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default="./index",
        help="Directory to save the index (default: ./index)"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="wilkertyl/bge-m3-portuguese-legal-v1",
        help="Path to the model"
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
        default="bill_id",
        help="Field name for document ID in JSON/CSV files (default: bill_id)"
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

    # Output file for model reference
    parser.add_argument(
        "--save_model_reference",
        action="store_true",
        default=True,
        help="Save model reference information to the index directory (default: True)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create index directory if it doesn't exist
    create_directory_if_not_exists(args.index_dir)

    # Determine which document source to use
    # By default, use the bills dataset URL if documents_path was not explicitly provided
    use_default_url = not any(arg in sys.argv for arg in ['--documents_path', '-documents_path'])

    # Handle document path or dataset download
    documents_path = args.documents_path
    if use_default_url or args.bills_dataset_url != "./bills_dataset.csv":
        print(f"Downloading bills dataset CSV from {args.bills_dataset_url}")
        download_path = os.path.join(args.index_dir, "downloaded_dataset")
        create_directory_if_not_exists(download_path)
        documents_path = download_dataset(args.bills_dataset_url, download_path)
        if not documents_path:
            print("Failed to download dataset. Exiting.")
            return

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)

    # Save model reference if requested
    if args.save_model_reference:
        model_info = {
            "model_path": args.model_path,
            "embedding_dim": args.embedding_dim,
            "max_length": args.chunk_size
        }
        model_info_path = os.path.join(args.index_dir, "model_reference.txt")
        with open(model_info_path, 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved model reference to {model_info_path}")

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

        # Save updated model reference even when using existing index
        if args.save_model_reference:
            print(f"Updating model reference for existing index")
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
        documents_path,
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