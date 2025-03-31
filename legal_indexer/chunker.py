"""
Document chunking functionality for legal documents
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
                r"CONCLUSÃO"
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
        # Tokenize the entire document
        tokens = self.tokenizer.encode(document, add_special_tokens=False)

        chunks = []
        chunk_id = 0

        # Find all section boundaries
        section_boundaries = self._find_section_boundaries(document, tokens)

        # Add chunking based on both token count and section boundaries
        start_idx = 0

        while start_idx < len(tokens):
            # Try to find a section boundary within an acceptable range
            end_idx = start_idx + self.max_chunk_size

            # Check if there's a section boundary we can use
            section_end = None
            for boundary in section_boundaries:
                if start_idx < boundary <= start_idx + self.max_chunk_size:
                    section_end = boundary

            # If we found a good section boundary, use it
            if section_end:
                end_idx = section_end

            # Make sure we don't exceed the document length
            end_idx = min(end_idx, len(tokens))

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

            # Make sure we're making progress
            if start_idx >= len(tokens):
                break

            if start_idx < 0:
                start_idx = 0

        return chunks

    def _find_section_boundaries(self, document: str, tokens: List[int]) -> List[int]:
        """
        Find section boundaries in the document

        Args:
            document: Text of the document
            tokens: List of token IDs

        Returns:
            List of token positions for section boundaries
        """
        section_boundaries = []
        lines = document.split("\n")
        current_pos = 0

        for line in lines:
            if self.section_regex.search(line):
                # Find the token position corresponding to this line
                line_tokens = self.tokenizer.encode(document[:current_pos + len(line)],
                                                    add_special_tokens=False)
                section_boundaries.append(len(line_tokens))
            current_pos += len(line) + 1  # +1 for the newline

        return section_boundaries