#!/usr/bin/env python3
"""
Script to search an existing index
"""
import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path so we can import the legal_indexer package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_indexer.chunker import DocumentChunker
from legal_indexer.embedder import DocumentEmbedder
from legal_indexer.indexer import LegalDocumentIndexer
from legal_indexer.utils import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Search a document index")

    # Required parameters
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory where the index is stored"
    )

    # Search parameters
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="BAAI/bge-m3",
        help="Path to the model (default: BAAI/bge-m3)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (default: auto-detect)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.query and not args.interactive:
        print("Please provide a query with --query or use --interactive mode")
        return

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)

    # Create components
    chunker = DocumentChunker(tokenizer=tokenizer)

    embedder = DocumentEmbedder(
        model=model,
        tokenizer=tokenizer,
        device=args.device
    )

    # Load the indexer
    indexer = LegalDocumentIndexer.load(
        directory=args.index_dir,
        chunker=chunker,
        embedder=embedder
    )

    if args.interactive:
        # Interactive search mode
        print("\nEntering interactive search mode. Type 'exit' to quit.\n")

        while True:
            query = input("Enter search query: ")

            if query.lower() in ["exit", "quit", "q"]:
                break

            if not query.strip():
                continue

            start_time = time.time()
            results = indexer.search(query, k=args.top_k)
            search_time = time.time() - start_time

            print(f"\nFound {len(results)} results in {search_time:.2f} seconds\n")

            for i, result in enumerate(results):
                print(f"Result {i + 1} (Score: {result['score']:.4f}):")
                print(f"Document: {result['doc_id']}, Chunk: {result['chunk_id']}")
                print(result['text'])
                print("-" * 80)

            print()
    else:
        # Single query mode
        start_time = time.time()
        results = indexer.search(args.query, k=args.top_k)
        search_time = time.time() - start_time

        print(f"\nSearch results for: {args.query}")
        print(f"Found {len(results)} results in {search_time:.2f} seconds\n")

        for i, result in enumerate(results):
            print(f"Result {i + 1} (Score: {result['score']:.4f}):")
            print(f"Document: {result['doc_id']}, Chunk: {result['chunk_id']}")
            print(result['text'])
            print("-" * 80)


if __name__ == "__main__":
    main()