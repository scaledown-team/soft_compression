"""
Dataset class for ScaleDown training.

Handles:
    - Query-document pairs with teacher labels
    - Memory token insertion
    - Reranking score labels (optional)
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ScaleDownDataset(Dataset):
    """
    Dataset for ScaleDown training following OSCAR paper format.

    Expected data format:
        {
            "query": str,
            "documents": List[str],  # Retrieved documents
            "answer": str,  # Teacher-generated answer
            "reranking_scores": Optional[List[float]],  # Teacher reranking scores
        }
    """

    def __init__(
        self,
        data: List[Dict],
        config,
        tokenizer: Optional[AutoTokenizer] = None,
        compressor_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.data = data
        self.config = config

        # Load tokenizers
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.generator_model_name)
        self.tokenizer = tokenizer

        # For ModernBERT compressor, we need a separate tokenizer
        if compressor_tokenizer is None and config.compressor_type == "modernbert":
            compressor_tokenizer = AutoTokenizer.from_pretrained(
                config.modernbert_model_name
            )
        self.compressor_tokenizer = compressor_tokenizer or tokenizer

        # Add special tokens for memory and reranking
        self.memory_tokens = [f"<MEM_{i}>" for i in range(config.num_memory_tokens)]
        self.reranking_token = "<RR>"

        # Add to tokenizer (if not already present)
        special_tokens = self.memory_tokens.copy()
        if config.enable_reranking:
            special_tokens.append(self.reranking_token)

        num_added = self.compressor_tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

        if num_added > 0:
            logger.info(f"Added {num_added} special tokens to tokenizer")

        logger.info(f"Dataset initialized with {len(data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training example.

        Returns a dictionary with:
            - query_input_ids: Tokenized query
            - query_attention_mask: Query mask
            - doc_input_ids: Tokenized documents with memory tokens
            - doc_attention_mask: Document mask
            - memory_token_positions: Positions of memory tokens
            - labels: Tokenized teacher answer
            - reranking_token_positions: Positions of RR token (if enabled)
            - reranking_labels: Teacher reranking scores (if enabled)
        """
        example = self.data[idx]

        query = example["query"]
        documents = example["documents"][: self.config.num_documents_train]
        answer = example["answer"]

        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Process each document
        doc_input_ids_list = []
        doc_attention_mask_list = []
        memory_positions_list = []
        rr_positions_list = []

        for doc in documents:
            # Create document with memory tokens
            # Format: [Document tokens] [MEM_1] [MEM_2] ... [MEM_l] [RR]?

            # Tokenize document (leave space for memory tokens)
            max_doc_len = (
                self.config.max_seq_length
                - self.config.num_memory_tokens
                - (1 if self.config.enable_reranking else 0)
            )

            doc_encoding = self.compressor_tokenizer(
                doc,
                max_length=max_doc_len,
                truncation=True,
                add_special_tokens=False,
            )

            # Get document token IDs
            doc_tokens = doc_encoding["input_ids"]

            # Add memory tokens
            mem_token_ids = [
                self.compressor_tokenizer.convert_tokens_to_ids(token)
                for token in self.memory_tokens
            ]

            # Combine: document + memory tokens + RR (if enabled)
            combined_tokens = doc_tokens + mem_token_ids

            # Track memory token positions
            mem_start = len(doc_tokens)
            mem_positions = list(range(mem_start, mem_start + self.config.num_memory_tokens))

            # Add RR token if reranking enabled
            rr_position = None
            if self.config.enable_reranking:
                rr_token_id = self.compressor_tokenizer.convert_tokens_to_ids(
                    self.reranking_token
                )
                combined_tokens.append(rr_token_id)
                rr_position = len(combined_tokens) - 1

            # Pad to max length
            padding_length = self.config.max_seq_length - len(combined_tokens)
            combined_tokens += [self.compressor_tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * (len(combined_tokens) - padding_length) + [0] * padding_length

            doc_input_ids_list.append(combined_tokens[: self.config.max_seq_length])
            doc_attention_mask_list.append(attention_mask[: self.config.max_seq_length])
            memory_positions_list.append(mem_positions)

            if rr_position is not None:
                rr_positions_list.append(rr_position)

        # Tokenize answer (labels for generation)
        answer_encoding = self.tokenizer(
            answer,
            max_length=self.config.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare output
        output = {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "doc_input_ids": torch.tensor(doc_input_ids_list),
            "doc_attention_mask": torch.tensor(doc_attention_mask_list),
            "memory_token_positions": torch.tensor(memory_positions_list),
            "labels": answer_encoding["input_ids"].squeeze(0),
        }

        # Add reranking data if enabled
        if self.config.enable_reranking:
            output["reranking_token_positions"] = torch.tensor(rr_positions_list)

            if "reranking_scores" in example:
                output["reranking_labels"] = torch.tensor(
                    example["reranking_scores"][: self.config.num_documents_train],
                    dtype=torch.float32,
                )

        return output


def collate_fn(batch):
    """Custom collate function to handle variable-length documents."""
    # All items should have the same structure, just stack them
    keys = batch[0].keys()

    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])

    return collated
