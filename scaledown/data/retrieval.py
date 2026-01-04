"""
Document retrieval utilities for ScaleDown dataset generation.

This module implements the retrieval pipeline from the OSCAR paper:
- SPLADE-v3 sparse retrieval
- Wikipedia-KILT corpus
- Retrieves top-K documents per query
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np


class SPLADERetriever:
    """
    SPLADE-v3 retriever for sparse document retrieval.

    Following OSCAR paper Section 4.1:
    - Uses SPLADE-v3 (naver/splade-cocondenser-ensembledistil)
    - Retrieves from Wikipedia-KILT corpus
    - Returns top-K documents per query
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize SPLADE retriever.

        Args:
            model_name: HuggingFace model name for SPLADE
            device: Device to run retrieval on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode queries into SPLADE sparse vectors.

        Args:
            queries: List of query strings
            batch_size: Batch size for encoding

        Returns:
            Sparse query representations (num_queries, vocab_size)
        """
        all_representations = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            # Get SPLADE representations
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # SPLADE: log(1 + ReLU(logits))
                # Max pooling over tokens
                representations = torch.max(
                    torch.log1p(torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
                    dim=1
                ).values

            all_representations.append(representations.cpu().numpy())

        return np.vstack(all_representations)

    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode documents into SPLADE sparse vectors.

        Args:
            documents: List of document strings
            batch_size: Batch size for encoding

        Returns:
            Sparse document representations (num_docs, vocab_size)
        """
        # Same encoding as queries for SPLADE
        return self.encode_queries(documents, batch_size)

    def retrieve(
        self,
        queries: List[str],
        document_corpus: List[Dict[str, str]],
        top_k: int = 20,
        batch_size: int = 32,
    ) -> List[List[Dict[str, str]]]:
        """
        Retrieve top-K documents for each query.

        Args:
            queries: List of query strings
            document_corpus: List of documents with 'id', 'title', 'text' fields
            top_k: Number of documents to retrieve per query
            batch_size: Batch size for encoding

        Returns:
            List of retrieved documents for each query
        """
        print(f"Encoding {len(queries)} queries...")
        query_reps = self.encode_queries(queries, batch_size)

        print(f"Encoding {len(document_corpus)} documents...")
        doc_texts = [f"{doc['title']} {doc['text']}" for doc in document_corpus]
        doc_reps = self.encode_documents(doc_texts, batch_size)

        print("Computing similarities...")
        # Sparse dot product (SPLADE uses sparse representations)
        similarities = query_reps @ doc_reps.T

        print("Retrieving top-K documents...")
        results = []
        for i, query in enumerate(queries):
            # Get top-K document indices
            top_indices = np.argsort(similarities[i])[-top_k:][::-1]

            retrieved_docs = []
            for idx in top_indices:
                doc = document_corpus[idx].copy()
                doc['retrieval_score'] = float(similarities[i, idx])
                retrieved_docs.append(doc)

            results.append(retrieved_docs)

        return results


class WikipediaKILTCorpus:
    """
    Wikipedia-KILT corpus loader.

    The OSCAR paper uses Wikipedia-KILT as the document corpus.
    Download from: https://github.com/facebookresearch/KILT
    """

    def __init__(self, corpus_path: Optional[str] = None):
        """
        Initialize corpus loader.

        Args:
            corpus_path: Path to Wikipedia-KILT corpus file
        """
        self.corpus_path = corpus_path
        self.documents = []

    def load_corpus(self, max_documents: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load Wikipedia-KILT corpus.

        Args:
            max_documents: Maximum number of documents to load (for testing)

        Returns:
            List of documents with 'id', 'title', 'text' fields
        """
        if self.corpus_path is None:
            raise ValueError(
                "corpus_path not provided. Download Wikipedia-KILT from:\n"
                "https://github.com/facebookresearch/KILT\n"
                "wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
            )

        import json

        print(f"Loading Wikipedia-KILT corpus from {self.corpus_path}...")
        self.documents = []

        with open(self.corpus_path, 'r') as f:
            for i, line in enumerate(f):
                if max_documents and i >= max_documents:
                    break

                doc = json.loads(line)
                self.documents.append({
                    'id': doc['wikipedia_id'],
                    'title': doc['wikipedia_title'],
                    'text': doc['text'][0] if doc['text'] else "",  # First paragraph
                })

        print(f"Loaded {len(self.documents)} documents")
        return self.documents

    def get_documents(self) -> List[Dict[str, str]]:
        """Get loaded documents."""
        if not self.documents:
            raise ValueError("Corpus not loaded. Call load_corpus() first.")
        return self.documents


def retrieve_documents_for_queries(
    queries: List[str],
    corpus_path: Optional[str] = None,
    top_k: int = 20,
    max_corpus_size: Optional[int] = None,
) -> List[Dict[str, any]]:
    """
    Main function to retrieve documents for queries using SPLADE-v3.

    This follows the OSCAR paper's retrieval pipeline:
    1. Load Wikipedia-KILT corpus
    2. Encode queries and documents with SPLADE-v3
    3. Retrieve top-K documents per query

    Args:
        queries: List of query strings
        corpus_path: Path to Wikipedia-KILT corpus
        top_k: Number of documents to retrieve per query
        max_corpus_size: Maximum corpus size (for testing)

    Returns:
        List of dictionaries with 'query' and 'documents' fields
    """
    # Load corpus
    corpus = WikipediaKILTCorpus(corpus_path)
    documents = corpus.load_corpus(max_documents=max_corpus_size)

    # Initialize retriever
    retriever = SPLADERetriever()

    # Retrieve documents
    retrieved = retriever.retrieve(queries, documents, top_k=top_k)

    # Format results
    results = []
    for query, docs in zip(queries, retrieved):
        results.append({
            'query': query,
            'documents': [
                {
                    'id': doc['id'],
                    'title': doc['title'],
                    'text': doc['text'],
                    'retrieval_score': doc['retrieval_score'],
                }
                for doc in docs
            ]
        })

    return results
