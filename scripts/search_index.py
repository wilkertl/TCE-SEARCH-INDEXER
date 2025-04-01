#!/usr/bin/env python3
"""
Script to search an existing index with max pooling for chunks and model reference
"""
import argparse
import time
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path so we can import the legal_indexer package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_indexer.chunker import DocumentChunker
from legal_indexer.embedder import DocumentEmbedder
from legal_indexer.indexer import LegalDocumentIndexer
from legal_indexer.utils import load_model_and_tokenizer


def get_model_from_reference(index_dir):
    """
    Get model path from the model reference file if it exists
    """
    reference_path = os.path.join(index_dir, "model_reference.txt")
    if os.path.exists(reference_path):
        model_info = {}
        with open(reference_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    model_info[key.strip()] = value.strip()

        if "model_path" in model_info:
            print(f"Using model from reference: {model_info['model_path']}")
            return model_info["model_path"]

    return None


def max_pooling_results(results, top_k=5):
    """
    Perform max pooling on search results to return the highest score for each document
    """
    # Group results by document ID
    doc_results = defaultdict(list)
    for result in results:
        doc_results[result['doc_id']].append(result)

    # For each document, keep only the chunk with the highest score
    max_results = []
    for doc_id, chunks in doc_results.items():
        max_chunk = max(chunks, key=lambda x: x['score'])
        max_results.append(max_chunk)

    # Sort by score and take top_k
    max_results.sort(key=lambda x: x['score'], reverse=True)
    return max_results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Search a document index")

    # Required parameters
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./index",
        help="Directory where the index is stored (default: ./index)"
    )

    # Search parameters
    # Define your default query here - replace with your actual query
    DEFAULT_QUERY = "Programa nacional de alimentacao escolar "

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=f"Search query (default: '{DEFAULT_QUERY}')"
    )

    # Define your default top_k here
    DEFAULT_TOP_K = 10  # Change this to your preferred number

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive search mode"
    )

    parser.add_argument(
        "--chunk_results",
        type=int,
        default=20,
        help="Number of chunk results to use for max pooling (default: 20)"
    )

    parser.add_argument(
        "--max_pooling",
        action="store_true",
        default=True,
        help="Use max pooling to return the highest score per document (default: True)"
    )

    # Model parameters
    # Define your default model here
    DEFAULT_MODEL = "wilkertyl/bge-m3-portuguese-legal-v1"  # Replace with your preferred model

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,  # Now uses your specified default model
        help=f"Path to the model (default: {DEFAULT_MODEL}, overrides model reference if specified)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (default: auto-detect)"
    )

    parser.add_argument(
        "--save_search_log",
        action="store_true",
        default=False,
        help="Save search queries and results to a log file (default: False)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Use the model from reference if enabled
    # Even with a default model set, we can optionally prefer the reference model
    USE_REFERENCE_MODEL_IF_AVAILABLE = True  # Set to False if you always want to use your default model

    if USE_REFERENCE_MODEL_IF_AVAILABLE:
        referenced_model = get_model_from_reference(args.index_dir)
        if referenced_model:
            print(f"Using model from reference: {referenced_model}")
            print(f"(Overriding default model: {args.model_path})")
            args.model_path = referenced_model

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

    # Initialize search log if enabled
    search_log = []

    def execute_search(query, top_k, chunk_results, use_max_pooling):
        """Helper function to execute search with consistent logic"""
        start_time = time.time()

        # First, get a larger number of chunk results for max pooling
        all_results = indexer.search(query, k=chunk_results)

        # Apply max pooling if enabled
        if use_max_pooling:
            results = max_pooling_results(all_results, top_k=top_k)
            result_type = "document"
        else:
            results = all_results[:top_k]
            result_type = "chunk"

        search_time = time.time() - start_time

        # Log search if enabled
        if args.save_search_log:
            search_log.append({
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results_count": len(results),
                "search_time_seconds": search_time,
                "max_pooling_used": use_max_pooling,
                "top_results": [
                    {"doc_id": r["doc_id"], "score": r["score"]}
                    for r in results[:3]  # Log only top 3 for brevity
                ]
            })

        return results, search_time, result_type

        # By default, run in non-interactive mode (remove the interactive check)
        # Single query mode
        results, search_time, result_type = execute_search(
            args.query, args.top_k, args.chunk_results, args.max_pooling
        )

        print(f"\nSearch results for: {args.query}")
        print(f"Found {len(results)} {result_type} results in {search_time:.2f} seconds")
        print(f"Using model: {args.model_path}")
        print(f"Max pooling: {'ON' if args.max_pooling else 'OFF'}\n")

        for i, result in enumerate(results):
            print(f"Result {i + 1} (Score: {result['score']:.4f}):")
            print(f"Document: {result['doc_id']}, Chunk: {result['chunk_id']}")
            print(result['text'])
            print("-" * 80)

    # Save search log if enabled
    if args.save_search_log and search_log:
        log_path = os.path.join(args.index_dir, "search_log.json")
        existing_log = []

        # Load existing log if it exists
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    existing_log = json.load(f)
                except json.JSONDecodeError:
                    existing_log = []

        # Append new searches and save
        with open(log_path, 'w') as f:
            json.dump(existing_log + search_log, f, indent=2)

        print(f"Search log saved to {log_path}")


# Set RUN_AUTOMATICALLY to True to run without needing to call from command line
RUN_AUTOMATICALLY = True

if __name__ == "__main__":
    if RUN_AUTOMATICALLY:
        # This will run the script automatically with the default parameters
        sys.argv = [sys.argv[0]]  # Keep just the script name, ignore any actual command line args
        main()
    else:
        # Normal command-line execution
        main()