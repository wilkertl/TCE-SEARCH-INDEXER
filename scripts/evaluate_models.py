#!/usr/bin/env python3
"""
Script para avaliar modelos de recuperação de informação legal
"""
import os
import sys
from pathlib import Path
from typing import List
import argparse
import time
import pandas as pd
from collections import defaultdict
from search_index import max_pooling_results, load_csv_metadata, extract_metadata, get_model_from_reference

# Adiciona o diretório pai ao path para importar o pacote legal_indexer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_indexer.chunker import DocumentChunker
from legal_indexer.embedder import DocumentEmbedder
from legal_indexer.indexer import LegalDocumentIndexer
from legal_indexer.utils import load_model_and_tokenizer
from avaliacao import evaluate, qrel_from_ulysses
from models import *

def search_and_format_results(indexer, query: str, query_id: str, csv_path, doc_metadata, top_k: int = 30) -> PytrecModelRankedQueries:

    """
    Executa uma busca e formata os resultados para avaliação
    """
    results = indexer.search(query, k=top_k)
    results = max_pooling_results(results, 15)
    
    # Converter resultados para o formato PytrecModelRankedQueries
    documents = []
    for result in results:

        doc_id = result['doc_id']

        metadata = extract_metadata(doc_id, doc_metadata)

        documents.append(
            DocumentRankedByModel(
                id=metadata['name'],
                score=result['score']
            )
        )
    
    return PytrecModelRankedQueries(
        id=query_id,
        documents=documents
    )

def evaluate_model(model_path: str, index_dir: str, qrel_path: str, dataset_path, output_file: str = None):
    """
    Avalia um modelo usando o conjunto de testes Ulysses
    
    Args:
        model_path: Caminho para o modelo
        index_dir: Diretório do índice FAISS
        qrel_path: Caminho para o arquivo de avaliação (fold_teste.csv)
        output_file: Arquivo para salvar os resultados (opcional)
    """
    # Carregar o qrel (conjunto de avaliação)
    qrel = qrel_from_ulysses(qrel_path)
    # Carregar modelo e tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    doc_metadata = load_csv_metadata(dataset_path)

    # Criar componentes
    chunker = DocumentChunker(tokenizer=tokenizer)
    embedder = DocumentEmbedder(model=model, tokenizer=tokenizer)
    
    # Carregar o indexador
    indexer = LegalDocumentIndexer.load(
        directory=index_dir,
        chunker=chunker,
        embedder=embedder
    )
    
    # Executar buscas para todas as consultas no qrel
    model_results = []
    for query in qrel:
        # Usar o ID da consulta como texto de busca (simplificado - ajuste conforme necessário)
        search_results = search_and_format_results(indexer, query.id, query.id, dataset_path, doc_metadata)
        print(f"\n{search_results}\n")
        model_results.append(search_results)
    
    # Métricas a serem avaliadas
    metrics=["ndcg_cut_10", "map_cut_10", "recall_100"]
    
    # Executar avaliação
    evaluation = evaluate(qrel=qrel, model_results=model_results, metrics=metrics)
    
    # Exibir resultados
    print("\nResultados da Avaliação:")
    print("=" * 50)
    for metric, value in evaluation["mean_metric_results"].items():
        print(f"{metric:>15}: {value:.4f}")
    
    # Salvar resultados se especificado
    if output_file:
        with open(output_file, 'w') as f:
            import json
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"\nResultados salvos em {output_file}")

def main():

    DEFAULT_QUERY = "proibição de nomeação em cargos públicos de pessoas condenadas pelo crime de estupro e pela Lei Maria da Penha."

    parser = argparse.ArgumentParser(description="Search a document index")

    parser.add_argument(
        "--index_dir",
        type=str,
        default="./index_v4",
        help="Directory where the index is stored (default: ./index_v4)"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="./bills_dataset.csv",
        help="Path to the bills_dataset.csv for metadata (default: ./bills_dataset.csv)"
    )

    # Adicione estes novos argumentos:
    parser.add_argument(
        "--qrel_path",
        type=str,
        required=True,
        help="Path to the qrel file (e.g., fold_teste.csv)"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results (default: evaluation_results.json)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
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
            args.model_path = "wilkertyl/bge-m3-portuguese-legal-v4"
    
    print("data loaded")

    # Load document metadata from original CSV

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

    start_time = time.time()

    evaluate_model(
        model_path=args.model_path,
        index_dir=args.index_dir,
        qrel_path=args.qrel_path,
        dataset_path= args.csv_path,
        output_file=args.output_file
    )
    
    print(f"\nTempo total de execução: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()