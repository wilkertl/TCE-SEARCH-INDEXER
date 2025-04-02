import re
from typing import List, Dict, Any, Optional, Tuple
import unicodedata


class SimpleDocumentChunker:
    """Simple chunker for legislative documents using natural boundaries"""

    def __init__(
            self,
            tokenizer,
            max_chunk_size: int = 512,
            chunk_overlap: int = 50,
            min_chunk_size: int = 100
    ):
        self.tokenizer = tokenizer
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Patterns for cleaning document noise
        self.noise_patterns = [
            r"\*CD\d+\*",  # Digital signature codes
            r"Assinado eletronicamente.*",  # Signature verification lines
            r"Para verificar a assinatura.*",  # Verification URLs
            r"IN\s+C\s+n\.\d+\/\d+.*?Mesa",  # Page markers
            r"Ap\s*re\s*se\s*nt\s*aç\s*ão:\s*\d+\/\d+\/\d+.*"  # Presentation dates in footers
        ]
        self.noise_regex = re.compile("|".join(self.noise_patterns))

        # Common sentence-ending punctuation in Portuguese legal documents
        self.sentence_endings = r'[.!?:;]'
        self.sentence_boundary_regex = re.compile(f'({self.sentence_endings}\\s+)')

    def preprocess_document(self, document: str) -> str:
        """Clean and normalize document text"""
        # Replace multiple spaces and normalize whitespace
        doc = re.sub(r'\s+', ' ', document)

        # Remove digital signatures and verification lines
        doc = self.noise_regex.sub('', doc)

        # Normalize Unicode (convert accented characters to canonical form)
        doc = unicodedata.normalize('NFC', doc)

        return doc.strip()

    def extract_metadata(self, document: str) -> Dict[str, Any]:
        """Extract key metadata from the document"""
        metadata = {
            "doc_type": None,
            "doc_number": None,
            "doc_year": None,
            "doc_author": None,
            "doc_subject": None
        }

        # Extract document type and number
        doc_type_match = re.search(
            r"(PROJETO DE LEI|INDICAÇÃO|REQUERIMENTO|PROPOSTA)\s+N[º°]?\s*([\d\.]+)[,\/]?\s*(?:DE\s+(\d{4}))?",
            document, re.IGNORECASE
        )
        if doc_type_match:
            metadata["doc_type"] = doc_type_match.group(1)
            metadata["doc_number"] = doc_type_match.group(2)
            metadata["doc_year"] = doc_type_match.group(3) if doc_type_match.group(3) else None

        # Extract author
        author_match = re.search(r"\(Do\s+(?:Sr|Sra)\.?\s+([^)]+)\)", document, re.IGNORECASE)
        if author_match:
            metadata["doc_author"] = author_match.group(1).strip()

        # Extract subject (usually after "Requer" or similar verbs)
        subject_match = re.search(r"Requer(?:\s+ao\s+[^.]+)?\s+([^.]+)", document, re.IGNORECASE)
        if subject_match:
            metadata["doc_subject"] = subject_match.group(1).strip()

        return metadata

    def find_natural_boundaries(self, text: str) -> List[int]:
        """Find natural boundaries like sentence endings and paragraph breaks"""
        boundaries = []

        # Find paragraph boundaries (double line breaks)
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.end())

        # Find sentence boundaries
        for match in self.sentence_boundary_regex.finditer(text):
            boundaries.append(match.end())

        # Add specific legal document boundaries
        for match in re.finditer(r'(Art(?:igo)?\.?\s+\d+[º°]?|\n§\s+\d+[º°]?|Parágrafo\s+[úÚ]nico)', text):
            boundaries.append(match.start())

        # Sort boundaries by position
        return sorted(set(boundaries))

    def chunk_document(self, document: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on natural boundaries

        Args:
            document: The text of the document
            doc_id: Unique document identifier

        Returns:
            List of dictionaries containing chunk information
        """
        # Preprocess and clean document
        clean_doc = self.preprocess_document(document)

        # Extract metadata
        metadata = self.extract_metadata(clean_doc)

        # Find natural boundaries
        boundaries = self.find_natural_boundaries(clean_doc)

        # If document is empty, return empty list
        if not clean_doc:
            return []

        # Special handling for very short documents
        if len(clean_doc) < self.max_chunk_size * 1.5:
            chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-0",
                "text": clean_doc,
                "metadata": metadata
            }
            return [chunk]

        # Create chunks using natural boundaries
        chunks = []
        chunk_id = 0
        start_pos = 0

        while start_pos < len(clean_doc):
            # Find the furthest natural boundary within max_chunk_size
            end_pos = start_pos + self.max_chunk_size

            # Adjust to nearest natural boundary if possible
            if end_pos < len(clean_doc):
                # Find closest boundary before max_chunk_size
                valid_boundaries = [b for b in boundaries if start_pos < b < end_pos]

                if valid_boundaries and valid_boundaries[-1] - start_pos >= self.min_chunk_size:
                    end_pos = valid_boundaries[-1]
                else:
                    # If no valid boundary found, find next sentence end
                    next_sentence_end = clean_doc.find('.', end_pos)
                    if next_sentence_end != -1 and next_sentence_end - start_pos < self.max_chunk_size * 1.5:
                        end_pos = next_sentence_end + 1
            else:
                end_pos = len(clean_doc)

            # Create chunk
            chunk_text = clean_doc[start_pos:end_pos].strip()
            if chunk_text:
                chunk = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-{chunk_id}",
                    "text": chunk_text,
                    "metadata": metadata,
                    "start_pos": start_pos,
                    "end_pos": end_pos
                }
                chunks.append(chunk)
                chunk_id += 1

            # Move to next chunk with overlap
            start_pos = max(0, end_pos - self.chunk_overlap)

            # Make sure we're making forward progress
            if start_pos >= len(clean_doc) or end_pos == len(clean_doc):
                break

        # Post-process to ensure no tiny chunks at the end
        if chunks and len(chunks[-1]["text"]) < self.min_chunk_size:
            if len(chunks) > 1:
                # Merge the last tiny chunk with the previous one
                chunks[-2]["text"] += " " + chunks[-1]["text"]
                chunks[-2]["end_pos"] = chunks[-1]["end_pos"]
                chunks.pop()

        return chunks