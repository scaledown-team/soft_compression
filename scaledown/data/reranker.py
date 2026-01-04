"""
Reranking utilities for ScaleDown dataset generation.

This module implements the reranking pipeline from the OSCAR paper:
- DeBERTa-v3 cross-encoder for scoring query-document pairs
- Optional reranking scores for joint compression + reranking training
"""

import torch
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


class DeBERTaReranker:
    """
    DeBERTa-v3 reranker for query-document scoring.

    Following OSCAR paper Section 3.3:
    - Uses DeBERTa-v3 cross-encoder
    - Scores query-document pairs
    - Provides supervision for joint compression + reranking
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Can upgrade to DeBERTa-v3
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run reranking on

        Note: The paper uses DeBERTa-v3. Common alternatives:
        - "cross-encoder/ms-marco-MiniLM-L-6-v2" (faster, smaller)
        - "cross-encoder/ms-marco-deberta-v3-base" (closer to paper)
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def score_pairs(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Score query-document pairs.

        Args:
            query: Query string
            documents: List of document strings
            batch_size: Batch size for scoring

        Returns:
            Relevance scores (num_documents,)
        """
        scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # Create query-document pairs
            pairs = [[query, doc] for doc in batch_docs]

            # Tokenize
            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Score
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)

            scores.append(logits.cpu().numpy())

        return np.concatenate(scores)

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, str]],
        batch_size: int = 16,
    ) -> List[Dict[str, str]]:
        """
        Rerank documents by relevance score.

        Args:
            query: Query string
            documents: List of document dicts with 'text' field
            batch_size: Batch size for scoring

        Returns:
            Documents sorted by relevance with added 'reranking_score' field
        """
        # Extract document texts
        doc_texts = [doc['text'] for doc in documents]

        # Score documents
        scores = self.score_pairs(query, doc_texts, batch_size)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['reranking_score'] = float(score)

        # Sort by score (descending)
        reranked = sorted(documents, key=lambda x: x['reranking_score'], reverse=True)

        return reranked


def add_reranking_scores(
    queries_with_docs: List[Dict[str, any]],
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 16,
) -> List[Dict[str, any]]:
    """
    Add reranking scores to retrieved documents.

    This follows the OSCAR paper's optional reranking approach:
    1. Score each query-document pair with cross-encoder
    2. Add scores as supervision for joint compression + reranking
    3. Student learns to predict these scores with RR token

    Args:
        queries_with_docs: List of dicts with 'query' and 'documents' fields
        reranker_model_name: HuggingFace model name for cross-encoder
        batch_size: Batch size for scoring

    Returns:
        Input list with 'reranking_scores' field added to each item
    """
    # Initialize reranker
    print(f"Loading reranker model: {reranker_model_name}...")
    reranker = DeBERTaReranker(model_name=reranker_model_name)

    results = []

    for i, item in enumerate(queries_with_docs):
        print(f"Reranking documents {i+1}/{len(queries_with_docs)}...")

        # Get documents
        documents = item['documents']

        # Score documents
        doc_texts = [doc['text'] for doc in documents]
        scores = reranker.score_pairs(item['query'], doc_texts, batch_size)

        # Add scores to result
        result = item.copy()
        result['reranking_scores'] = scores.tolist()

        # Also add individual scores to documents
        for doc, score in zip(result['documents'], scores):
            doc['reranking_score'] = float(score)

        results.append(result)

    print("Reranking complete!")
    return results


def rerank_and_filter(
    queries_with_docs: List[Dict[str, any]],
    top_k: int = 5,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, any]]:
    """
    Rerank documents and keep only top-K.

    This is useful to filter retrieved documents before teacher generation:
    1. Retrieve 20 documents with sparse retrieval (SPLADE)
    2. Rerank with cross-encoder
    3. Keep top-5 for answer generation (reduces noise)

    Args:
        queries_with_docs: List of dicts with 'query' and 'documents' fields
        top_k: Number of top documents to keep
        reranker_model_name: HuggingFace model name for cross-encoder

    Returns:
        Input list with documents filtered to top-K by reranking score
    """
    # Add reranking scores
    results = add_reranking_scores(queries_with_docs, reranker_model_name)

    # Filter to top-K
    for item in results:
        # Sort documents by reranking score
        sorted_docs = sorted(
            item['documents'],
            key=lambda x: x['reranking_score'],
            reverse=True
        )

        # Keep top-K
        item['documents'] = sorted_docs[:top_k]

        # Update reranking_scores to match
        item['reranking_scores'] = [
            doc['reranking_score'] for doc in item['documents']
        ]

    return results
