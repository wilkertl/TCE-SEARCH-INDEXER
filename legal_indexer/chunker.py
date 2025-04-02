import re
from typing import List, Dict, Any, Optional, Tuple
import unicodedata

class LegislativeDocumentChunker:
    """Specialized chunker for legislative documents"""

    def __init__(
            self,
            tokenizer,
            max_chunk_size: int = 512,
            chunk_overlap: int = 100,  # Increased overlap
            min_chunk_size: int = 100   # Minimum chunk size
    ):
        self.tokenizer = tokenizer
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Define section hierarchies with importance weights
        self.section_patterns = {
            # Major document types - highest priority
            "document_type": {
                "patterns": [
                    r"PROJETO DE LEI(?:\s+COMPLEMENTAR)?\s+N[º°]?\s*[\d\.]+[,\/]?\s*(?:DE\s+\d{4})?",
                    r"INDICAÇÃO\s+N[º°]?\s*[\d\.]+[,\/]?\s*(?:DE\s+\d{4})?",
                    r"REQUERIMENTO\s+N[º°]?\s*[\d\.]+[,\/]?\s*(?:DE\s+\d{4})?",
                    r"PROPOSTA DE EMENDA CONSTITUCIONAL\s+N[º°]?\s*[\d\.]+[,\/]?\s*(?:DE\s+\d{4})?"
                ],
                "weight": 30
            },
            # Main structural elements - high priority
            "main_structure": {
                "patterns": [
                    r"EMENTA",
                    r"JUSTIFICAÇÃO",
                    r"CONCLUSÃO",
                    r"PARECER"
                ],
                "weight": 25
            },
            # Major divisions - medium-high priority
            "major_division": {
                "patterns": [
                    r"TÍTULO\s+[IVXLCDM]+",
                    r"CAPÍTULO\s+[IVXLCDM]+"
                ],
                "weight": 20
            },
            # Legal articles - medium priority
            "articles": {
                "patterns": [
                    r"Art(?:igo)?\.\s+\d+[º°]?",
                    r"ARTIGO\s+\d+[º°]?"
                ],
                "weight": 15
            },
            # Subsections - medium-low priority
            "subsections": {
                "patterns": [
                    r"SEÇÃO\s+[IVXLCDM]+",
                    r"Parágrafo\s+[úÚ]nico",
                    r"§\s+\d+[º°]?"
                ],
                "weight": 10
            },
            # Minor divisions - low priority
            "minor_divisions": {
                "patterns": [
                    r"inciso\s+[IVXLCDM]+",
                    r"alínea\s+[a-z]",
                    r"item\s+\d+"
                ],
                "weight": 5
            }
        }

        # Compile all patterns
        self.all_patterns = {}
        for section_type, info in self.section_patterns.items():
            for pattern in info["patterns"]:
                self.all_patterns[pattern] = info["weight"]

        # Compile a single regex for efficiency
        pattern_strings = "|".join(f"({p})" for p in self.all_patterns.keys())
        self.section_regex = re.compile(pattern_strings, re.IGNORECASE)

        # Patterns for cleaning document noise
        self.noise_patterns = [
            r"\*CD\d+\*",  # Digital signature codes
            r"Assinado eletronicamente.*",  # Signature verification lines
            r"Para verificar a assinatura.*",  # Verification URLs
            r"IN\s+C\s+n\.\d+\/\d+.*?Mesa",  # Page markers
            r"Ap\s*re\s*se\s*nt\s*aç\s*ão:\s*\d+\/\d+\/\d+.*"  # Presentation dates in footers
        ]
        self.noise_regex = re.compile("|".join(self.noise_patterns))

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

    def find_weighted_boundaries(self, document: str) -> List[Tuple[int, int]]:
        """Find section boundaries with their importance weights"""
        boundaries = []

        # Process by lines for better boundary detection
        lines = document.split("\n")
        char_position = 0

        for line in lines:
            line = line.strip()
            if not line:
                char_position += 1  # Account for newline
                continue

            # Check for section headers
            matches = list(self.section_regex.finditer(line))
            if matches:
                for match in matches:
                    # Get the pattern that matched
                    matched_text = match.group(0)
                    # Find which pattern matched
                    for pattern, weight in self.all_patterns.items():
                        if re.search(pattern, matched_text, re.IGNORECASE):
                            boundaries.append((char_position + match.start(), weight))
                            break

            char_position += len(line) + 1  # +1 for newline

        return boundaries

    def chunk_document(self, document: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split a document into overlapping chunks with metadata

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

        # Tokenize the entire document
        tokens = self.tokenizer.encode(clean_doc, add_special_tokens=False)

        # If document is empty, return empty list
        if not tokens:
            return []

        # Find section boundaries with weights
        weighted_boundaries = self.find_weighted_boundaries(clean_doc)

        # Convert character boundaries to token positions
        token_boundaries = []
        if weighted_boundaries:
            char_to_token = {}

            # Create mapping of character positions to token positions
            for i, token_id in enumerate(tokens):
                token_text = self.tokenizer.decode([token_id])
                char_len = len(token_text)

                # Map each character position to token position
                for j in range(char_len):
                    curr_pos = sum(len(self.tokenizer.decode([tid])) for tid in tokens[:i]) + j
                    char_to_token[curr_pos] = i

            # Convert character boundaries to token boundaries
            for char_pos, weight in weighted_boundaries:
                if char_pos in char_to_token:
                    token_pos = char_to_token[char_pos]
                    token_boundaries.append((token_pos, weight))
                else:
                    # Find closest mapping
                    closest_pos = min(char_to_token.keys(), key=lambda x: abs(x - char_pos))
                    token_boundaries.append((char_to_token[closest_pos], weight))

        # Create chunks
        chunks = []
        chunk_id = 0
        start_idx = 0

        # Special handling for short documents
        if len(tokens) <= self.max_chunk_size * 1.5:
            # For very short documents, keep as a single chunk
            chunk_text = clean_doc
            chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{chunk_id}",
                "text": chunk_text,
                "start_idx": 0,
                "end_idx": len(tokens),
                "metadata": metadata
            }
            chunks.append(chunk)
            return chunks

        # Process normal-sized documents with intelligent chunking
        while start_idx < len(tokens):
            # Calculate maximum possible end based on chunk size
            max_end_idx = min(start_idx + self.max_chunk_size, len(tokens))

            # Default end index
            end_idx = max_end_idx

            # Check for better section boundaries
            best_boundary = None
            best_weight = -1

            for token_pos, weight in token_boundaries:
                # Only consider boundaries within our chunk range and not too close to start
                if (start_idx < token_pos < max_end_idx and
                    token_pos - start_idx >= self.min_chunk_size):

                    # Prioritize higher weight boundaries
                    if weight > best_weight:
                        best_boundary = token_pos
                        best_weight = weight

            # Use the best boundary if found
            if best_boundary is not None:
                end_idx = best_boundary

            # Create the chunk
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Create chunk with metadata
            chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-{chunk_id}",
                "text": chunk_text,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "metadata": metadata,
                "boundary_weight": best_weight if best_weight > -1 else 0
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

        # Post-process chunks to handle important sections
        enhanced_chunks = self._enhance_chunks_with_context(chunks, clean_doc)

        return enhanced_chunks

    def _enhance_chunks_with_context(self, chunks, document):
        """Add important context to chunks and handle key sections"""
        if not chunks:
            return chunks

        # For key sections (like the main request in an INDICAÇÃO),
        # ensure they're not split across chunks
        bold_paragraphs = re.finditer(r"((?:Desta forma|Assim|Portanto)[^.]+(?:é necessário|solicito|requer)[^.]+\.)", document)

        for match in bold_paragraphs:
            bold_text = match.group(1)
            bold_start_char = match.start()
            bold_end_char = match.end()

            # Find which chunks contain parts of this important text
            containing_chunks = []
            for i, chunk in enumerate(chunks):
                if bold_text in chunk["text"]:
                    # Already contained in one chunk, no need to modify
                    containing_chunks = []
                    break

                # Check if chunk contains part of the bold text
                chunk_start_in_doc = document.find(chunk["text"])
                chunk_end_in_doc = chunk_start_in_doc + len(chunk["text"])

                if (chunk_start_in_doc <= bold_start_char < chunk_end_in_doc or
                    chunk_start_in_doc < bold_end_char <= chunk_end_in_doc or
                    bold_start_char <= chunk_start_in_doc < bold_end_char):
                    containing_chunks.append(i)

            # If the important text is split across chunks, create a special chunk
            if len(containing_chunks) > 1:
                first_chunk_idx = min(containing_chunks)
                last_chunk_idx = max(containing_chunks)

                # Create a new chunk with the important text and some context
                context_before = document[max(0, bold_start_char - 200):bold_start_char]
                context_after = document[bold_end_char:min(len(document), bold_end_char + 200)]

                special_chunk_text = context_before + bold_text + context_after
                special_chunk = {
                    "doc_id": chunks[0]["doc_id"],
                    "chunk_id": f"{chunks[0]['doc_id']}-special-{first_chunk_idx}-{last_chunk_idx}",
                    "text": special_chunk_text,
                    "metadata": chunks[0]["metadata"],
                    "is_key_section": True,
                    "boundary_weight": 30  # Highest priority
                }

                # Add this special chunk
                chunks.append(special_chunk)

        return chunks