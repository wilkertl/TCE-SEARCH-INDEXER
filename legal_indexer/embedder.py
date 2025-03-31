"""
Document embedding functionality
"""
import torch
import numpy as np
import faiss
from typing import List, Dict, Any, Union, Optional


class DocumentEmbedder:
    """Class for creating embeddings from document chunks"""

    def __init__(
            self,
            model,
            tokenizer,
            max_length: int = 512,
            device: Optional[str] = None,
            batch_size: int = 8
    ):
        """
        Initialize the document embedder

        Args:
            model: The model to use for encoding
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            device: Device to use for computation
            batch_size: Batch size for encoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode a list of document chunks into embeddings

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            Array of embeddings
        """
        texts = [chunk["text"] for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings

        Args:
            texts: List of texts to encode

        Returns:
            Array of embeddings
        """
        embeddings = []

        # Process in batches to avoid OOM
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings from model
                outputs = self.model(**inputs)

                # Use CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding

        Args:
            query: The query text

        Returns:
            Query embedding
        """
        # Use single-text embedding
        return self.embed_texts([query])

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for cosine similarity

        Args:
            embeddings: Array of embeddings

        Returns:
            Normalized embeddings
        """
        faiss.normalize_L2(embeddings)
        return embeddings