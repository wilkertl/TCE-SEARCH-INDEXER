#!/usr/bin/env python3
"""
Script to search an existing index with formatted output including document name from original CSV
"""
import argparse
import time
import sys
import os
import json
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add parent directory to path so we can import the legal_indexer package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_indexer.chunker import LegislativeDocumentChunker
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


def improved_max_pooling_results(results, top_k=20):
    """
    Improved max pooling that considers multiple chunks per document
    and also applies a minimum score threshold
    """
    # Group results by document ID
    doc_results = defaultdict(list)
    for result in results:
        doc_results[result['doc_id'].split('-')[0]].append(result)

    # For each document, calculate a combined score
    combined_results = []
    for doc_id, chunks in doc_results.items():
        # Sort chunks by score in descending order
        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
        # Use the highest score, but also consider other high-scoring chunks
        main_score = sorted_chunks[0]['score']
        # Add a bonus for documents with multiple high-scoring chunks
        bonus = sum(chunk['score'] for chunk in sorted_chunks[1:3]) * 0.1 if len(sorted_chunks) > 1 else 0

        combined_results.append({
            'doc_id': doc_id,
            'score': main_score + bonus,
            'top_chunk': sorted_chunks[0]['text'] if 'text' in sorted_chunks[0] else ""
        })

    # Sort by combined score and take top_k
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    return combined_results[:top_k]


def load_csv_metadata(csv_path):
    """
    Load document metadata from the original CSV file
    """
    doc_metadata = {}

    if os.path.exists(csv_path):
        try:
            # Try using pandas which handles various CSV formats better
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            if 'code' in df.columns and 'name' in df.columns:
                # Create a dictionary with code as key and name as value
                for _, row in df.iterrows():
                    doc_metadata[str(row['code'])] = {
                        'name': row['name'],
                        # Include other columns if needed
                        'sig_tipo': row.get('sig_tipo', ''),
                        'txt_ementa': row.get('txt_ementa', ''),
                        'em_tramitacao': row.get('em_tramitacao', ''),
                        'situacao': row.get('situacao', '')
                    }
                print(f"Loaded metadata for {len(doc_metadata)} documents from CSV")
            else:
                print(f"Warning: CSV does not contain required columns. Found: {', '.join(df.columns)}")
        except Exception as e:
            print(f"Warning: Could not load CSV metadata: {e}")

    return doc_metadata


def extract_metadata(doc_id, doc_metadata=None):
    """
    Extract code and name from document ID and available metadata
    """
    try:
        clean_doc_id = doc_id
        if '-' in doc_id:
            # If it's a chunk ID, get just the document part
            clean_doc_id = doc_id.split('-')[0]

        # Try to get the actual metadata from document metadata if available
        if doc_metadata and clean_doc_id in doc_metadata:
            metadata_entry = doc_metadata[clean_doc_id]
            if isinstance(metadata_entry, dict):
                return {
                    'code': int(clean_doc_id),
                    'name': metadata_entry.get('name', f"INC {clean_doc_id}"),
                    'sig_tipo': metadata_entry.get('sig_tipo', ''),
                    'em_tramitacao': metadata_entry.get('em_tramitacao', ''),
                    'situacao': metadata_entry.get('situacao', '')
                }

        # Default values if no metadata found
        return {
            'code': int(clean_doc_id) if clean_doc_id.isdigit() else clean_doc_id,
            'name': f"INC {clean_doc_id}"
        }
    except Exception as e:
        print(f"Warning: Error extracting metadata for {doc_id}: {e}")
        # Fallback if we can't parse the document ID
        return {'code': doc_id, 'name': f"Document {doc_id}"}

def main():
    # Define your default query here - replace with your actual query
    DEFAULT_QUERY = "Elaborar PLP alterando as atribuições do Conselho Monetário Nacional na Lei nº 4595/64."

    parser = argparse.ArgumentParser(description="Search a document index")

    parser.add_argument(
        "--index_dir",
        type=str,
        default="./index_v4",
        help="Directory where the index is stored (default: ./index)"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="./bills_dataset.csv",
        help="Path to the original bills_dataset.csv for metadata (default: ./bills_dataset.csv)"
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
        default=30,
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
            args.model_path = "wilkertyl/bge-m3-portuguese-legal-v4"

    print("data loaded")

    # Load document metadata from original CSV
    doc_metadata = load_csv_metadata(args.csv_path)

    # Load model and tokenizer (silently)
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)

    # Create components
    chunker = LegislativeDocumentChunker(tokenizer=tokenizer)
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
        metadata = extract_metadata(doc_id, doc_metadata)
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