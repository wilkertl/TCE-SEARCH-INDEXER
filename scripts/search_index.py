#!/usr/bin/env python3
"""
<<<<<<< Updated upstream
Script to search an existing index with formatted output including document name from original CSV
=======
Script to search documents with embedded model and direct database access
>>>>>>> Stashed changes
"""
import argparse
import time
import sys
import os
<<<<<<< Updated upstream
import json
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
=======
from pathlib import Path
import torch
import faiss
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
>>>>>>> Stashed changes


class DirectDocumentSearch:
    def __init__(self, index_dir, model_path, device=None):
        """
        Initialize the direct document search

        Args:
            index_dir: Directory where the index and metadata are stored
            model_path: Path to the model for embeddings
            device: Device to use for computation
        """
        self.index_dir = Path(index_dir)

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load index
        self.load_index()

    def load_index(self):
        """Load FAISS index and metadata"""
        # Load FAISS index
        index_path = self.index_dir / "faiss_index.bin"
        print(f"Loading index from: {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Load document metadata
        docs_path = self.index_dir / "documents.json"
        print(f"Loading document metadata from: {docs_path}")
        with open(docs_path, 'r') as f:
            self.documents = json.load(f)

        # Load chunk metadata
        chunks_path = self.index_dir / "chunks.json"
        print(f"Loading chunk metadata from: {chunks_path}")
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)

        # Create chunk-to-document mapping
        self.chunk_to_doc = {
            int(chunk_id): doc_id
            for doc_id, doc in self.documents.items()
            for chunk_id in doc.get("chunks", [])
        }

        print(f"Loaded {len(self.documents)} documents with {len(self.chunks)} chunks")

    def embed_query(self, query):
        """Create embedding for query text"""
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token (assuming BERT-like model)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)
        return embedding

    def search_chunks(self, query, k=10):
        """Search for chunks similar to query"""
        # Create query embedding
        query_embedding = self.embed_query(query)

        # Search the index
        distances, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx == -1 or distance == 0:  # Skip empty results
                continue

            chunk_id = str(idx)
            doc_id = self.chunk_to_doc.get(idx, "unknown")

            # Get chunk text
            chunk_text = self.chunks.get(chunk_id, {}).get("text", "")

            results.append({
                "score": float(distance),
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })

        return results

    def search_documents(self, query, top_k=5):
        """Search and return document-level results"""
        # Get more chunks to ensure good document coverage
        chunk_results = self.search_chunks(query, k=top_k * 3)

        # Aggregate by document
        doc_scores = {}
        doc_chunks = {}

        for result in chunk_results:
            doc_id = result['doc_id']
            score = result['score']

            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_chunks[doc_id] = []

            doc_chunks[doc_id].append(result)

            # Use max score for document ranking
            doc_scores[doc_id] = max(doc_scores[doc_id], score)

        # Create document results
        doc_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            # Get document metadata
            doc_metadata = self.documents.get(doc_id, {})
            doc_title = doc_metadata.get("title", f"Document {doc_id}")

            # Get best chunk
            best_chunk = max(doc_chunks[doc_id], key=lambda x: x['score'])

            doc_results.append({
                'doc_id': doc_id,
                'title': doc_title,
                'score': score,
                'top_chunk': best_chunk['text'],
                'chunk_count': len(doc_chunks[doc_id]),
                'chunks': doc_chunks[doc_id],
                'metadata': doc_metadata
            })

        # Return top results
        return doc_results[:top_k]


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

        # Default values
        code = int(clean_doc_id)
        name = f"INC {clean_doc_id}"  # Default format if no metadata found

        # Try to get the actual name from document metadata if available
        if doc_metadata and clean_doc_id in doc_metadata:
            metadata_entry = doc_metadata[clean_doc_id]
            if isinstance(metadata_entry, dict) and 'name' in metadata_entry:
                name = metadata_entry['name']

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
        default="./index_v3",
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
        default=5,
        help="Number of results to return (default: 5)"
    )

    parser.add_argument(
        "--chunk_results",
        type=int,
        default=20,
        help="Number of chunk results to use for max pooling (default: 20)"
    )

<<<<<<< Updated upstream
=======
    parser.add_argument(
        "--document_view",
        action="store_true",
        help="Return document-level results instead of chunks"
    )

    # Model parameters
>>>>>>> Stashed changes
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
            args.model_path = "wilkertyl/bge-m3-portuguese-legal-v3"

<<<<<<< Updated upstream
    print("data loaded")

    # Load document metadata from original CSV
    doc_metadata = load_csv_metadata(args.csv_path)

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
=======
    # Create search engine
    search_engine = DirectDocumentSearch(
        index_dir=args.index_dir,
        model_path=args.model_path,
        device=args.device
    )

    if args.interactive:
        # Interactive search mode
        print("\nEntering interactive search mode. Type 'exit' to quit.\n")
>>>>>>> Stashed changes

    # Execute search and measure time
    start_time = time.time()

    # Get chunk results
    all_results = indexer.search(args.query, k=args.chunk_results)

    # Apply max pooling to get document results
    results = max_pooling_results(all_results, top_k=args.top_k)

<<<<<<< Updated upstream
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
=======
            start_time = time.time()

            if args.document_view:
                results = search_engine.search_documents(query, top_k=args.top_k)
                print(f"\nFound {len(results)} document results in {time.time() - start_time:.2f} seconds\n")

                for i, result in enumerate(results):
                    print(f"Document {i + 1} (Score: {result['score']:.4f}):")
                    print(f"Title: {result['title']}")
                    print(f"Document ID: {result['doc_id']}")
                    print(f"Matching chunks: {result['chunk_count']}")
                    print(f"Top chunk excerpt: {result['top_chunk'][:200]}...")
                    print("-" * 80)
            else:
                results = search_engine.search_chunks(query, k=args.top_k)
                print(f"\nFound {len(results)} chunk results in {time.time() - start_time:.2f} seconds\n")

                for i, result in enumerate(results):
                    print(f"Result {i + 1} (Score: {result['score']:.4f}):")
                    print(f"Document: {result['doc_id']}, Chunk: {result['chunk_id']}")
                    print(result['text'])
                    print("-" * 80)

            print()
    else:
        # Single query mode
        start_time = time.time()

        if args.document_view:
            results = search_engine.search_documents(query=args.query, top_k=args.top_k)
            print(f"\nDocument search results for: {args.query}")
            print(f"Found {len(results)} document results in {time.time() - start_time:.2f} seconds\n")

            for i, result in enumerate(results):
                print(f"Document {i + 1} (Score: {result['score']:.4f}):")
                print(f"Title: {result['title']}")
                print(f"Document ID: {result['doc_id']}")
                print(f"Matching chunks: {result['chunk_count']}")
                print(f"Top chunk excerpt: {result['top_chunk'][:200]}...")
                print("-" * 80)
        else:
            results = search_engine.search_chunks(query=args.query, k=args.top_k)
            print(f"\nChunk search results for: {args.query}")
            print(f"Found {len(results)} results in {time.time() - start_time:.2f} seconds\n")

            for i, result in enumerate(results):
                print(f"Result {i + 1} (Score: {result['score']:.4f}):")
                print(f"Document: {result['doc_id']}, Chunk: {result['chunk_id']}")
                print(result['text'])
                print("-" * 80)

>>>>>>> Stashed changes

if __name__ == "__main__":
    if RUN_AUTOMATICALLY:
        # This will run the script automatically with the default parameters
        sys.argv = [sys.argv[0]]  # Keep just the script name, ignore any actual command line args
        main()
    else:
        # Normal command-line execution
        main()