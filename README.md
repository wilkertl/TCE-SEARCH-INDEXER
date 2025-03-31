# Legal Document Indexer

A Python library for indexing and searching legal documents using FAISS and transformer-based embeddings.

## Features

- Intelligent chunking of legal documents respecting section boundaries
- High-quality embeddings using BGE-M3 or your fine-tuned model
- Fast vector search with FAISS
- Incremental document processing
- Support for various document formats (JSON, CSV, text files)
- Command-line interface for common operations
- Modular design for easy extension

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/legal-document-indexer.git
cd legal-document-indexer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Build an index

```bash
python scripts/build_index.py \
  --documents_path data/legal_documents.json \
  --index_dir indexes/legal_index \
  --model_path /path/to/your/finetuned/model
```

### 2. Search the index

```bash
python scripts/search_index.py \
  --index_dir indexes/legal_index \
  --query "Quais são as diretrizes para gerenciamento de resíduos hospitalares?" \
  --model_path /path/to/your/finetuned/model \
  --top_k 5
```

### 3. Interactive search mode

```bash
python scripts/search_index.py \
  --index_dir indexes/legal_index \
  --interactive \
  --model_path /path/to/your/finetuned/model
```

### 4. Using the main script

```bash
# Build index and search in one command
python main.py \
  --mode both \
  --documents_path data/legal_documents.json \
  --index_dir indexes/legal_index \
  --model_path /path/to/your/finetuned/model \
  --query "Quais são as diretrizes para gerenciamento de resíduos hospitalares?"

# Use with a config file
python main.py --config config/my_config.json
```

## Configuration

You can provide parameters via command-line arguments or a configuration file. See `config/default_config.json` for an example.

## Project Structure

```
legal_indexer/
├── __init__.py
├── chunker.py        # Document chunking functionality
├── embedder.py       # Embedding generation
├── indexer.py        # FAISS indexing and searching
├── utils.py          # Utility functions
└── data_loaders.py   # Document loading from various sources

scripts/
├── build_index.py    # Script to build index
└── search_index.py   # Script to search existing index

config/
└── default_config.json  # Default configuration

main.py               # Main entry point script
```

## Using Multiple Models

If you want to create and compare indices with different embedding models:

```bash
# Index with model A
python scripts/build_index.py \
  --documents_path data/legal_documents.json \
  --index_dir indexes/model_a_index \
  --model_path /path/to/model_a

# Index with model B
python scripts/build_index.py \
  --documents_path data/legal_documents.json \
  --index_dir indexes/model_b_index \
  --model_path /path/to/model_b
```

## Working with Large Document Collections

For very large document collections, consider:

1. Using the `--index_type ivf` option for faster approximate search
2. Processing documents in batches with incremental saves
3. Using a powerful GPU for faster embedding generation

## License

[MIT License](LICENSE)