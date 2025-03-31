"""
Utility functions for the legal document indexer
"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Tuple, Optional


def load_model_and_tokenizer(
        model_path: str,
        device: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from path

    Args:
        model_path: Path or name of the model
        device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model {model_path} on {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    return model, tokenizer


def get_embedding_dim(model: Any) -> int:
    """
    Get the embedding dimension from a model

    Args:
        model: The model to get the embedding dimension from

    Returns:
        Embedding dimension
    """
    # Try to get from config
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size

    # BGE-M3 has 1024 dimensions
    if 'bge-m3' in str(model.__class__).lower():
        return 1024

    # Default to 768 (common for many models)
    return 768


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a JSON file

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration to
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_directory_if_not_exists(directory: str):
    """
    Create a directory if it doesn't exist

    Args:
        directory: Directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"