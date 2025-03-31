"""
Data loading functions for the legal document indexer
"""
import os
import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_documents_from_json(
    file_path: str,
    id_field: str = 'id',
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> Dict[str, str]:
    """
    Load documents from a JSON file

    Args:
        file_path: Path to the JSON file
        id_field: Field to use as document ID
        text_field: Field to use as document text
        encoding: File encoding

    Returns:
        Dictionary mapping document IDs to document texts
    """
    documents = {}

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Handle nested structures
                    doc_id = None
                    doc_text = None

                    # Try to find id_field and text_field at any level
                    if id_field in item and text_field in item:
                        doc_id = str(item[id_field])
                        doc_text = item[text_field]
                    else:
                        # Look for fields in nested dicts
                        for key, value in item.items():
                            if isinstance(value, dict):
                                if id_field in value and text_field in value:
                                    doc_id = str(value[id_field])
                                    doc_text = value[text_field]
                                    break

                    if doc_id and doc_text:
                        documents[doc_id] = doc_text
        elif isinstance(data, dict):
            # Case 1: Simple {id1: text1, id2: text2} format
            if all(isinstance(v, str) for v in data.values()):
                documents = {str(k): v for k, v in data.items()}
            # Case 2: {key1: {id_field: id1, text_field: text1}, ...} format
            elif all(isinstance(v, dict) for v in data.values()):
                for key, value in data.items():
                    if id_field in value and text_field in value:
                        doc_id = str(value[id_field])
                        documents[doc_id] = value[text_field]
            # Case 3: {id_field: id1, text_field: [text1, text2, ...]} format (single doc with multiple texts)
            elif id_field in data and text_field in data and isinstance(data[text_field], list):
                doc_id = str(data[id_field])
                documents[doc_id] = " ".join(data[text_field])
            # Case 4: {id_field: id1, text_field: text1} format (single document)
            elif id_field in data and text_field in data:
                doc_id = str(data[id_field])
                documents[doc_id] = data[text_field]

        logger.info(f"Loaded {len(documents)} documents from JSON file: {file_path}")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading documents from JSON ({file_path}): {e}")

    return documents

def load_documents_from_csv(
    file_path: str,
    id_column: str = 'id',
    text_column: str = 'text',
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> Dict[str, str]:
    """
    Load documents from a CSV file

    Args:
        file_path: Path to the CSV file
        id_column: Column to use as document ID
        text_column: Column to use as document text
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        Dictionary mapping document IDs to document texts
    """
    documents = {}

    try:
        # Try using pandas for better handling of complex CSVs
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, error_bad_lines=False)

        if id_column in df.columns and text_column in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row[id_column]) and pd.notna(row[text_column]):
                    doc_id = str(row[id_column])
                    documents[doc_id] = str(row[text_column])

        logger.info(f"Loaded {len(documents)} documents from CSV file: {file_path}")
    except Exception as e:
        logger.error(f"Error loading documents from CSV with pandas: {e}")

        # Fallback to standard csv module
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    if id_column in row and text_column in row:
                        doc_id = str(row[id_column])
                        documents[doc_id] = row[text_column]
            logger.info(f"Loaded {len(documents)} documents from CSV file using csv module: {file_path}")
        except Exception as e2:
            logger.error(f"Error loading documents from CSV with csv module: {e2}")

    return documents

def load_documents_from_directory(
    directory: str,
    extensions: List[str] = ['.txt'],
    recursive: bool = True,
    encoding: str = 'utf-8'
) -> Dict[str, str]:
    """
    Load documents from text files in a directory

    Args:
        directory: Directory containing text files
        extensions: List of file extensions to include
        recursive: Whether to search subdirectories
        encoding: File encoding

    Returns:
        Dictionary mapping document IDs (filenames) to document texts
    """
    documents = {}

    try:
        for root, _, files in os.walk(directory):
            if not recursive and root != directory:
                continue

            for file in files:
                # Check if file has the right extension
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)

                    # Use filename without extension as document ID
                    doc_id = os.path.splitext(file)[0]

                    # Read file content
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()

                        documents[doc_id] = content
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from directory: {directory}")
    except Exception as e:
        logger.error(f"Error loading documents from directory: {e}")

    return documents

def load_documents_from_jsonl(
    file_path: str,
    id_field: str = 'id',
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> Dict[str, str]:
    """
    Load documents from a JSONL file (one JSON object per line)

    Args:
        file_path: Path to the JSONL file
        id_field: Field to use as document ID
        text_field: Field to use as document text
        encoding: File encoding

    Returns:
        Dictionary mapping document IDs to document texts
    """
    documents = {}

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            line_num = 0
            for line in f:
                line_num += 1
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and id_field in data and text_field in data:
                        doc_id = str(data[id_field])
                        documents[doc_id] = data[text_field]
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents from JSONL file: {file_path}")
    except Exception as e:
        logger.error(f"Error loading documents from JSONL file: {e}")

    return documents

def detect_file_type(file_path: str) -> str:
    """
    Detect the type of a file based on its extension

    Args:
        file_path: Path to the file

    Returns:
        File type ('json', 'jsonl', 'csv', 'txt', 'directory', 'unknown')
    """
    if os.path.isdir(file_path):
        return 'directory'

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.json':
        return 'json'
    elif ext == '.jsonl':
        return 'jsonl'
    elif ext == '.csv':
        return 'csv'
    elif ext in ['.txt', '.md', '.rst', '.text']:
        return 'txt'
    else:
        return 'unknown'

def load_documents(
    file_path: str,
    id_field: str = 'id',
    text_field: str = 'text',
    encoding: str = 'utf-8',
    **kwargs
) -> Dict[str, str]:
    """
    Load documents from a file, automatically detecting the file type

    Args:
        file_path: Path to the file
        id_field: Field to use as document ID
        text_field: Field to use as document text
        encoding: File encoding
        **kwargs: Additional arguments to pass to the appropriate loader

    Returns:
        Dictionary mapping document IDs to document texts
    """
    file_type = detect_file_type(file_path)
    logger.info(f"Detected file type: {file_type} for {file_path}")

    if file_type == 'json':
        return load_documents_from_json(file_path, id_field, text_field, encoding)
    elif file_type == 'jsonl':
        return load_documents_from_jsonl(file_path, id_field, text_field, encoding)
    elif file_type == 'csv':
        delimiter = kwargs.get('delimiter', ',')
        return load_documents_from_csv(file_path, id_field, text_field, delimiter, encoding)
    elif file_type == 'txt':
        # Single text file - use filename as ID
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return {doc_id: content}
        except Exception as e:
            logger.error(f"Error loading document from text file: {e}")
            return {}
    elif file_type == 'directory':
        extensions = kwargs.get('extensions', ['.txt'])
        recursive = kwargs.get('recursive', True)
        return load_documents_from_directory(file_path, extensions, recursive, encoding)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return {}

def combine_document_collections(*document_collections) -> Dict[str, str]:
    """
    Combine multiple document collections into one

    Args:
        *document_collections: Variable number of document dictionaries

    Returns:
        Combined dictionary of documents
    """
    combined = {}
    duplicate_count = 0

    for documents in document_collections:
        for doc_id, text in documents.items():
            if doc_id in combined:
                duplicate_count += 1
                # Append a suffix to make the ID unique
                doc_id = f"{doc_id}_duplicate_{duplicate_count}"
            combined[doc_id] = text

    return combined