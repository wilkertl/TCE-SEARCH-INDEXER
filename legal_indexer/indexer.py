"""
FAISS indexing and search functionality
"""
import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm

from .chunker import DocumentChunker
from .embedder import DocumentEmbedder


class DocumentIndex:
    """Class for indexing and searching document embeddings"""

    def __init__(
            self,
            embedding_dim: int = 1024,
            index_type: str = "flat"
    ):
        """
        Initialize the document index

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index to use ('flat', 'ivf', or 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.id_to_doc_map = {}
        self.processed_doc_ids = set()
        self.total_chunks = 0

    def create_index(self):
        """Create a new FAISS index based on index_type"""
        if self.index_type == "flat":
            # Simple flat index - exact search, good for up to ~1M vectors
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            # IVF index - approximate search, good for larger collections
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)  # 100 clusters
            # Note: IVF indices need training before use
        elif self.index_type == "hnsw":
            # HNSW index - very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def needs_training(self) -> bool:
        """Check if the index requires training"""
        return self.index_type == "ivf"

    def train_index(self, embeddings: np.ndarray):
        """
        Train the index with sample embeddings

        Args:
            embeddings: Sample embeddings for training
        """
        if self.needs_training():
            self.index.train(embeddings)

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add embeddings to the index

        Args:
            embeddings: Embeddings to add
            chunks: Corresponding chunk information
        """
        if self.index is None:
            self.create_index()

        # Train if needed and this is the first batch
        if self.needs_training() and self.total_chunks == 0:
            self.train_index(embeddings)

        # Add to index
        start_id = self.total_chunks
        self.index.add(embeddings)

        # Update mapping
        for i, chunk in enumerate(chunks):
            idx = start_id + i
            self.id_to_doc_map[idx] = {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            }

        # Update processed doc IDs and total count
        self.processed_doc_ids.update(set(chunk["doc_id"] for chunk in chunks))
        self.total_chunks += len(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Embedding of the query
            k: Number of results to return

        Returns:
            List of search results
        """
        if self.index is None or self.total_chunks == 0:
            return []

        # Search index
        scores, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 indicates no match found
                chunk_info = self.id_to_doc_map[idx]
                results.append({
                    "doc_id": chunk_info["doc_id"],
                    "chunk_id": chunk_info["chunk_id"],
                    "text": chunk_info["text"],
                    "score": float(score)
                })

        return results

    def save(self, directory: str):
        """
        Save the index and metadata

        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        # Save mapping and config
        with open(os.path.join(directory, "metadata.pickle"), "wb") as f:
            pickle.dump({
                "id_to_doc_map": self.id_to_doc_map,
                "total_chunks": self.total_chunks,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "processed_doc_ids": self.processed_doc_ids
            }, f)

    @classmethod
    def load(cls, directory: str) -> "DocumentIndex":
        """
        Load an existing index from a directory

        Args:
            directory: Directory to load from

        Returns:
            Loaded DocumentIndex
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.pickle"), "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        index = cls(
            embedding_dim=metadata["embedding_dim"],
            index_type=metadata.get("index_type", "flat")  # Default to flat for backward compatibility
        )

        # Load FAISS index
        index.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        # Restore metadata
        index.id_to_doc_map = metadata["id_to_doc_map"]
        index.total_chunks = metadata["total_chunks"]
        index.processed_doc_ids = metadata.get("processed_doc_ids", set())

        return index


class LegalDocumentIndexer:
    """Main class for the legal document indexing and retrieval system"""

    def __init__(
            self,
            chunker: DocumentChunker,
            embedder: DocumentEmbedder,
            index: Optional[DocumentIndex] = None,
            embedding_dim: int = 1024,
            index_type: str = "flat"
    ):
        """
        Initialize the legal document indexer

        Args:
            chunker: Document chunker instance
            embedder: Document embedder instance
            index: Document index instance (optional)
            embedding_dim: Embedding dimension (used if index is None)
            index_type: FAISS index type (used if index is None)
        """
        self.chunker = chunker
        self.embedder = embedder

        # Create index if not provided
        if index is None:
            self.index = DocumentIndex(embedding_dim=embedding_dim, index_type=index_type)
        else:
            self.index = index

    def process_documents(
            self,
            documents: Dict[str, str],
            batch_size: int = 100,
            index_dir: Optional[str] = None,
            show_progress: bool = True
    ):
        """
        Process, chunk, and index a dictionary of documents

        Args:
            documents: Dictionary with document IDs as keys and text as values
            batch_size: Number of documents to process in each batch
            index_dir: Directory to save the index incrementally (optional)
            show_progress: Whether to show progress bars
        """
        # Filter out documents that have already been processed
        new_docs = {doc_id: text for doc_id, text in documents.items()
                    if doc_id not in self.index.processed_doc_ids}

        if not new_docs:
            print("No new documents to process.")
            return

        if show_progress:
            print(f"Processing {len(new_docs)} new documents out of {len(documents)} total documents")

        # Process documents in batches
        doc_ids = list(new_docs.keys())

        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i:i + batch_size]
            batch_docs = {doc_id: new_docs[doc_id] for doc_id in batch_doc_ids}

            # 1. Chunk documents
            all_chunks = []
            iterator = tqdm(batch_docs.items(),
                            desc=f"Chunking (batch {i // batch_size + 1})") if show_progress else batch_docs.items()

            for doc_id, text in iterator:
                chunks = self.chunker.chunk_document(text, doc_id)
                all_chunks.extend(chunks)

            if not all_chunks:
                continue

            if show_progress:
                print(f"Created {len(all_chunks)} chunks from {len(batch_docs)} documents")

            # 2. Encode chunks
            if show_progress:
                print("Encoding chunks...")

            embeddings = self.embedder.embed_chunks(all_chunks)
            embeddings = self.embedder.normalize_embeddings(embeddings)

            # 3. Add to index
            self.index.add_embeddings(embeddings, all_chunks)

            # 4. Save incrementally if directory is provided
            if index_dir:
                self.index.save(index_dir)
                if show_progress:
                    print(f"Saved index incrementally (total chunks: {self.index.total_chunks})")

        if show_progress:
            print(f"Index now contains {self.index.total_chunks} total chunks")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of dictionaries with search results
        """
        # 1. Encode query
        query_embedding = self.embedder.embed_query(query)
        query_embedding = self.embedder.normalize_embeddings(query_embedding)

        # 2. Search index
        return self.index.search(query_embedding, k)

    def save(self, directory: str):
        """
        Save the index

        Args:
            directory: Directory to save to
        """
        self.index.save(directory)

    @classmethod
    def load(
            cls,
            directory: str,
            chunker: DocumentChunker,
            embedder: DocumentEmbedder
    ) -> "LegalDocumentIndexer":
        """
        Load an existing legal document indexer

        Args:
            directory: Directory to load from
            chunker: Document chunker instance
            embedder: Document embedder instance

        Returns:
            Loaded LegalDocumentIndexer
        """
        # Load index
        index = DocumentIndex.load(directory)

        # Create LegalDocumentIndexer with loaded index
        return cls(chunker=chunker, embedder=embedder, index=index)