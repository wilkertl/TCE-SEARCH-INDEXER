#!/usr/bin/env python3
"""
Script to search an existing index with formatted output
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


def extract_metadata(doc_id):
    """
    Extract code and name from document ID
    Format assumption: INC XXXX/YYYY
    """
    # Try to extract a document name if it looks like a legal document code
    try:
        if '-' in doc_id:
            # If it's a chunk ID, get just the document part
            doc_id = doc_id.split('-')[0]

        # Default values
        code = int(doc_id)
        name = f"INC {doc_id}"

        # You can enhance this with more sophisticated metadata extraction
        # if your document IDs have a different format

        return {'code': code, 'name': name}
    except:
        # Fallback if we can't parse the document ID
        return {'code': doc_id, 'name': doc_id}


def main():
    # Define your default query here - replace with your actual query
    DEFAULT_QUERY = "proibição de nomeação em cargos públicos de pessoas condenadas pelo crime de estupro e pela Lei Maria da Penha."

    parser = argparse.ArgumentParser(description="Search a document index")

    parser.add_argument(
        "--index_dir",
        type=str,
        default="./index",
        help="Directory where the index is stored (default: ./index)"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=f"Search query (default: '{DEFAULT_QUERY}')"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    parser.add_argument(
        "--chunk_results",
        type=int,
        default=20,
        help="Number of chunk results to use for max pooling (default: 20)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,  # Will use referenced model by default
        help="Path to the model (overrides model reference if specified)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (default: auto-detect)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Use model from reference if not explicitly provided
    if args.model_path is None:
        referenced_model = get_model_from_reference(args.index_dir)
        if referenced_model:
            args.model_path = referenced_model
        else:
            args.model_path = "wilkertyl/bge-m3-portuguese-legal-v1"

    print("data loaded")

    # Load model and tokenizer (silently)
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

    print("data indexed")
    print(f"query={args.query}")

    # Execute search and measure time
    start_time = time.time()

    # Get chunk results
    all_results = indexer.search(args.query, k=args.chunk_results)

    # Apply max pooling to get document results
    results = max_pooling_results(all_results, top_k=args.top_k)

    # Display results in the requested format
    for result in results:
        doc_id = result['doc_id']
        metadata = extract_metadata(doc_id)
        score = result['score']
        print(f"* metadata=[{{'code': {metadata['code']}, 'name': '{metadata['name']}'}}], score={score}")

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


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
