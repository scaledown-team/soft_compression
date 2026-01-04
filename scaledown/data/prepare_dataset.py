"""
Main dataset preparation script for ScaleDown.

This script implements the full data pipeline from the OSCAR paper:
1. Load queries (MS MARCO, PISCO, or custom)
2. Retrieve documents with SPLADE-v3 from Wikipedia-KILT
3. Rerank with DeBERTa-v3 cross-encoder (optional)
4. Generate answers with teacher LLM (Mistral-7B)
5. Save in ScaleDown training format

Usage:
    # Full pipeline with MS MARCO queries
    python -m scaledown.data.prepare_dataset \\
        --queries_file queries.txt \\
        --corpus_path kilt_knowledgesource.json \\
        --output_file train_data.json \\
        --enable_reranking

    # Quick test with synthetic queries
    python -m scaledown.data.prepare_dataset \\
        --num_synthetic_queries 100 \\
        --output_file test_data.json
"""

import argparse
import json
from typing import List, Dict, Optional
from pathlib import Path

from .retrieval import retrieve_documents_for_queries
from .teacher import generate_teacher_answers
from .reranker import add_reranking_scores, rerank_and_filter


def load_queries(
    queries_file: Optional[str] = None,
    num_synthetic_queries: Optional[int] = None,
) -> List[str]:
    """
    Load queries from file or generate synthetic queries.

    Args:
        queries_file: Path to file with one query per line
        num_synthetic_queries: Number of synthetic queries to generate

    Returns:
        List of query strings
    """
    if queries_file:
        print(f"Loading queries from {queries_file}...")
        with open(queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(queries)} queries")
        return queries

    elif num_synthetic_queries:
        print(f"Generating {num_synthetic_queries} synthetic queries...")
        # Generate simple synthetic queries for testing
        queries = [
            f"What is example topic {i}?"
            for i in range(num_synthetic_queries)
        ]
        return queries

    else:
        raise ValueError("Must provide either queries_file or num_synthetic_queries")


def download_ms_marco_queries(output_path: str) -> str:
    """
    Download MS MARCO dev queries (for reproduction).

    The OSCAR paper uses MS MARCO dev queries (6,980 queries).

    Args:
        output_path: Path to save queries

    Returns:
        Path to saved queries file
    """
    import requests

    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.dev.small.tsv"

    print(f"Downloading MS MARCO queries from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    queries = []
    for line in response.text.strip().split('\n'):
        qid, query = line.split('\t')
        queries.append(query)

    print(f"Downloaded {len(queries)} MS MARCO queries")

    # Save
    with open(output_path, 'w') as f:
        for query in queries:
            f.write(query + '\n')

    print(f"Saved to {output_path}")
    return output_path


def prepare_dataset(
    queries: List[str],
    corpus_path: Optional[str] = None,
    output_file: str = "train_data.json",
    top_k_retrieval: int = 20,
    top_k_reranking: Optional[int] = 5,
    enable_reranking: bool = False,
    teacher_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_corpus_size: Optional[int] = None,
    teacher_8bit: bool = False,
) -> str:
    """
    Main dataset preparation pipeline.

    Following the OSCAR paper's approach:
    1. Retrieve documents with SPLADE-v3
    2. Optionally rerank with cross-encoder
    3. Generate answers with teacher LLM
    4. Save in ScaleDown training format

    Args:
        queries: List of query strings
        corpus_path: Path to Wikipedia-KILT corpus
        output_file: Path to save training data
        top_k_retrieval: Number of documents to retrieve per query
        top_k_reranking: Number of documents to keep after reranking (None = keep all)
        enable_reranking: Add reranking scores for joint training
        teacher_model: Teacher LLM model name
        reranker_model: Reranker model name
        max_corpus_size: Maximum corpus size (for testing)
        teacher_8bit: Use 8-bit quantization for teacher

    Returns:
        Path to saved training data
    """
    print("=" * 80)
    print("ScaleDown Dataset Preparation")
    print("=" * 80)
    print(f"Queries: {len(queries)}")
    print(f"Top-K retrieval: {top_k_retrieval}")
    print(f"Top-K reranking: {top_k_reranking}")
    print(f"Enable reranking: {enable_reranking}")
    print(f"Teacher model: {teacher_model}")
    print(f"Reranker model: {reranker_model}")
    print("=" * 80)

    # Step 1: Retrieve documents
    print("\n[Step 1/3] Retrieving documents with SPLADE-v3...")
    queries_with_docs = retrieve_documents_for_queries(
        queries=queries,
        corpus_path=corpus_path,
        top_k=top_k_retrieval,
        max_corpus_size=max_corpus_size,
    )

    # Step 2: Rerank (optional)
    if top_k_reranking or enable_reranking:
        print("\n[Step 2/3] Reranking documents with cross-encoder...")

        if top_k_reranking:
            # Rerank and filter to top-K
            queries_with_docs = rerank_and_filter(
                queries_with_docs,
                top_k=top_k_reranking,
                reranker_model_name=reranker_model,
            )
        elif enable_reranking:
            # Just add scores (no filtering)
            queries_with_docs = add_reranking_scores(
                queries_with_docs,
                reranker_model_name=reranker_model,
            )
    else:
        print("\n[Step 2/3] Skipping reranking...")

    # Step 3: Generate teacher answers
    print("\n[Step 3/3] Generating answers with teacher LLM...")
    training_data = generate_teacher_answers(
        queries_with_docs,
        teacher_model_name=teacher_model,
        load_in_8bit=teacher_8bit,
    )

    # Step 4: Format and save
    print("\nFormatting training data...")
    formatted_data = []
    for item in training_data:
        formatted_item = {
            'query': item['query'],
            'documents': [doc['text'] for doc in item['documents']],
            'answer': item['answer'],
        }

        # Add reranking scores if available
        if enable_reranking and 'reranking_scores' in item:
            formatted_item['reranking_scores'] = item['reranking_scores']

        formatted_data.append(formatted_item)

    # Save
    print(f"\nSaving {len(formatted_data)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)

    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print(f"Saved to: {output_file}")
    print(f"Examples: {len(formatted_data)}")
    print("=" * 80)

    return output_file


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare ScaleDown training dataset following OSCAR paper"
    )

    # Query sources
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--queries_file",
        type=str,
        help="Path to file with queries (one per line)"
    )
    query_group.add_argument(
        "--num_synthetic_queries",
        type=int,
        help="Generate N synthetic queries for testing"
    )
    query_group.add_argument(
        "--download_ms_marco",
        action="store_true",
        help="Download MS MARCO dev queries"
    )

    # Corpus
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Path to Wikipedia-KILT corpus (kilt_knowledgesource.json)"
    )
    parser.add_argument(
        "--max_corpus_size",
        type=int,
        default=None,
        help="Maximum corpus size for testing"
    )

    # Retrieval
    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=20,
        help="Number of documents to retrieve per query"
    )

    # Reranking
    parser.add_argument(
        "--enable_reranking",
        action="store_true",
        help="Add reranking scores for joint compression + reranking"
    )
    parser.add_argument(
        "--top_k_reranking",
        type=int,
        default=None,
        help="Filter to top-K documents after reranking"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Reranker model name"
    )

    # Teacher
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Teacher LLM model name"
    )
    parser.add_argument(
        "--teacher_8bit",
        action="store_true",
        help="Use 8-bit quantization for teacher LLM"
    )

    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        default="train_data.json",
        help="Path to save training data"
    )

    args = parser.parse_args()

    # Load queries
    if args.download_ms_marco:
        queries_file = download_ms_marco_queries("ms_marco_queries.txt")
        queries = load_queries(queries_file=queries_file)
    else:
        queries = load_queries(
            queries_file=args.queries_file,
            num_synthetic_queries=args.num_synthetic_queries,
        )

    # Prepare dataset
    prepare_dataset(
        queries=queries,
        corpus_path=args.corpus_path,
        output_file=args.output_file,
        top_k_retrieval=args.top_k_retrieval,
        top_k_reranking=args.top_k_reranking,
        enable_reranking=args.enable_reranking,
        teacher_model=args.teacher_model,
        reranker_model=args.reranker_model,
        max_corpus_size=args.max_corpus_size,
        teacher_8bit=args.teacher_8bit,
    )


if __name__ == "__main__":
    main()
