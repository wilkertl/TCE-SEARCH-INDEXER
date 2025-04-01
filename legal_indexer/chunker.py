"""
Document chunking functionality for legal documents - Optimized version
"""
import re
from typing import List, Dict, Any, Optional


class DocumentChunker:
    """Class for chunking documents into smaller pieces"""

    def __init__(
            self,
            tokenizer,
            max_chunk_size: int = 512,
            chunk_overlap: int = 50,
            legal_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the document chunker

        Args:
            tokenizer: Tokenizer to use for determining token counts
            max_chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between consecutive chunks
            legal_patterns: List of regex patterns for legal document sections
        """
        self.tokenizer = tokenizer
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Default legal document section patterns if not provided
        if legal_patterns is None:
            self.legal_patterns = [
                r"ARTIGO \d+", r"Art\. \d+", r"CAPÍTULO \d+", r"TÍTULO \d+",
                r"SEÇÃO \d+", r"PARÁGRAFO \d+", r"§ \d+º", r"EMENTA", r"DISPOSITIVO",
                r"CONCLUSÃO", r"INDICAÇÃO Nº", r"REQUERIMENTO Nº"
            ]
        else:
            self.legal_patterns = legal_patterns

        self.section_regex = re.compile("|".join(self.legal_patterns))

    def chunk_document(self, document: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split a document into overlapping chunks

        Args:
            document: The text of the document
            doc_id: Unique document identifier

        Returns:
            List of dictionaries containing chunk information
        """
        # If document is empty, return empty list
        if not document or not document.strip():
            return []

        # Tokenize the entire document
        tokens = self.tokenizer.encode(document, add_special_tokens=False)

        # If no tokens, return empty list
        if not tokens:
            return []

        chunks = []
        chunk_id = 0

        # More efficient section boundary detection
        section_boundaries = self._find_section_boundaries_efficient(document)

        # Add chunking based on both token count and section boundaries
        start_idx = 0

        while start_idx < len(tokens):
            # Define the end index based on max chunk size
            end_idx = min(start_idx + self.max_chunk_size, len(tokens))

            # Check if there's a better section boundary to use for this chunk
            for boundary in section_boundaries:
                # Only consider boundaries within our chunk that aren't too close to start
                if (start_idx < boundary < end_idx and
                    boundary - start_idx > self.max_chunk_size // 4):  # Ensure meaningful chunks
                    end_idx = boundary
                    break

            # Decode this chunk back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Create chunk metadata
            chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{chunk_id}",
                "text": chunk_text,
                "start_idx": start_idx,
                "end_idx": end_idx
            }

            chunks.append(chunk)
            chunk_id += 1

            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap

            # Make sure we're making forward progress
            if start_idx >= len(tokens) or end_idx == len(tokens):
                break

            if start_idx < 0:
                start_idx = 0

        return chunks

    def _find_section_boundaries_efficient(self, document: str) -> List[int]:
        """
        Find section boundaries in the document more efficiently by working with character positions
        and converting to token positions only at the end.

        Args:
            document: Text of the document

        Returns:
            List of token positions for section boundaries
        """
        # Find character positions of section boundaries
        char_boundaries = []

        # Split document by lines
        lines = document.split("\n")
        char_position = 0

        for line in lines:
            # Check if this line matches any section pattern
            if self.section_regex.search(line):
                char_boundaries.append(char_position)

            # Move character position forward (include newline)
            char_position += len(line) + 1

        # Now convert character positions to token positions in a single pass
        token_boundaries = []
        if char_boundaries:
            # Create substrings up to each boundary and tokenize once
            for boundary in char_boundaries:
                boundary_tokens = self.tokenizer.encode(
                    document[:boundary],
                    add_special_tokens=False
                )
                token_boundaries.append(len(boundary_tokens))

        return token_boundaries