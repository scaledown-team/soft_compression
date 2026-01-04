"""
Main ScaleDown model that combines compressor and generator.

This module implements the full ScaleDown pipeline following the OSCAR paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging

from .compressor import NLayersCompressor, ModernBERTCompressor
from .generator import ScaleDownGenerator

logger = logging.getLogger(__name__)


class ScaleDownModel(nn.Module):
    """
    Complete ScaleDown model for RAG compression.

    Architecture (from OSCAR paper):
        1. Compressor: Maps (query, document) → compressed embeddings
           - Option A: First N layers of generator (n_layers)
           - Option B: ModernBERT encoder (modernbert) [our novel contribution]

        2. Generator: LLM with LoRA that generates answer from:
           - Original query tokens
           - Compressed document embeddings

    Training:
        - Loss: Distillation from teacher LLM + optional reranking loss
        - L(C, G) = -Σ log G(a_i | query, c_1, ..., c_k, a_{<i})
                    + λ Σ (r_i - r'_i)²
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        logger.info(f"Initializing ScaleDown model with {config.compressor_type} compressor")

        # Initialize generator first (needed for N-Layers compressor)
        self.generator = ScaleDownGenerator(config)

        # Get hidden sizes
        generator_hidden_size = self.generator.get_hidden_size()
        config.generator_hidden_size = generator_hidden_size

        # Initialize compressor based on type
        if config.compressor_type == "n_layers":
            logger.info(
                f"Creating N-Layers compressor with {config.num_compressor_layers} layers"
            )
            self.compressor = NLayersCompressor(config, self.generator.model)

        elif config.compressor_type == "modernbert":
            logger.info("Creating ModernBERT compressor")
            self.compressor = ModernBERTCompressor(config)

        else:
            raise ValueError(f"Unknown compressor_type: {config.compressor_type}")

        # Store configuration
        self.num_memory_tokens = config.num_memory_tokens
        self.enable_reranking = config.enable_reranking
        self.reranking_loss_weight = config.reranking_loss_weight

        logger.info(f"ScaleDown model initialized successfully")
        logger.info(f"  Compressor type: {config.compressor_type}")
        logger.info(f"  Memory tokens per doc: {config.num_memory_tokens}")
        logger.info(f"  Compression rate: {config.compression_rate}x")
        logger.info(f"  Reranking: {config.enable_reranking}")

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        memory_token_positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reranking_token_positions: Optional[torch.Tensor] = None,
        reranking_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ScaleDown model.

        Args:
            query_input_ids: [batch_size, query_len] - Query tokens
            query_attention_mask: [batch_size, query_len] - Query mask
            doc_input_ids: [batch_size, num_docs, doc_len] - Document tokens
                doc_len includes: [Document tokens] [MEM_1] ... [MEM_l] [RR]?
            doc_attention_mask: [batch_size, num_docs, doc_len] - Document mask
            memory_token_positions: [batch_size, num_docs, num_mem_tokens]
                - Positions of memory tokens in each document
            labels: [batch_size, answer_len] - Teacher-generated answer tokens
            reranking_token_positions: [batch_size, num_docs] - Position of RR token
            reranking_labels: [batch_size, num_docs] - Teacher reranking scores

        Returns:
            Dictionary with losses and outputs:
                - loss: Total loss (generation + optional reranking)
                - generation_loss: Cross-entropy loss for generation
                - reranking_loss: L2 loss for reranking scores (if enabled)
                - logits: Generator output logits
                - reranking_scores: Predicted reranking scores (if enabled)
        """
        batch_size, num_docs, doc_len = doc_input_ids.shape

        # Compress each document
        all_compressed_embeddings = []
        all_reranking_scores = []

        for doc_idx in range(num_docs):
            # Get current document
            doc_ids = doc_input_ids[:, doc_idx, :]  # [batch_size, doc_len]
            doc_mask = doc_attention_mask[:, doc_idx, :]  # [batch_size, doc_len]
            mem_positions = memory_token_positions[:, doc_idx, :]  # [batch_size, num_mem_tokens]

            # Get reranking token position if enabled
            rr_position = None
            if self.enable_reranking and reranking_token_positions is not None:
                rr_position = reranking_token_positions[:, doc_idx]  # [batch_size]

            # Compress document
            compressed_emb, rr_score = self.compressor(
                input_ids=doc_ids,
                attention_mask=doc_mask,
                memory_token_positions=mem_positions,
                reranking_token_position=rr_position,
            )
            # compressed_emb: [batch_size, num_mem_tokens, hidden_size]
            # rr_score: [batch_size] or None

            all_compressed_embeddings.append(compressed_emb)
            if rr_score is not None:
                all_reranking_scores.append(rr_score)

        # Concatenate all compressed embeddings
        compressed_embeddings = torch.cat(all_compressed_embeddings, dim=1)
        # [batch_size, num_docs * num_mem_tokens, hidden_size]

        # Concatenate reranking scores if enabled
        reranking_scores = None
        if all_reranking_scores:
            reranking_scores = torch.stack(all_reranking_scores, dim=1)
            # [batch_size, num_docs]

        # Generate answer
        gen_outputs = self.generator(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            compressed_embeddings=compressed_embeddings,
            labels=labels,
            return_dict=True,
        )

        # Compute losses
        generation_loss = gen_outputs["loss"]
        total_loss = generation_loss

        reranking_loss = None
        if self.enable_reranking and reranking_labels is not None:
            # L2 loss between predicted and teacher scores
            reranking_loss = torch.nn.functional.mse_loss(
                reranking_scores, reranking_labels
            )
            total_loss = total_loss + self.reranking_loss_weight * reranking_loss

        return {
            "loss": total_loss,
            "generation_loss": generation_loss,
            "reranking_loss": reranking_loss,
            "logits": gen_outputs["logits"],
            "reranking_scores": reranking_scores,
        }

    def generate(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        memory_token_positions: torch.Tensor,
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate answers for queries with retrieved documents.

        Args:
            query_input_ids: [batch_size, query_len]
            query_attention_mask: [batch_size, query_len]
            doc_input_ids: [batch_size, num_docs, doc_len]
            doc_attention_mask: [batch_size, num_docs, doc_len]
            memory_token_positions: [batch_size, num_docs, num_mem_tokens]
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            generated_ids: [batch_size, generated_len]
        """
        batch_size, num_docs, doc_len = doc_input_ids.shape

        # Compress all documents
        all_compressed_embeddings = []

        for doc_idx in range(num_docs):
            doc_ids = doc_input_ids[:, doc_idx, :]
            doc_mask = doc_attention_mask[:, doc_idx, :]
            mem_positions = memory_token_positions[:, doc_idx, :]

            compressed_emb, _ = self.compressor(
                input_ids=doc_ids,
                attention_mask=doc_mask,
                memory_token_positions=mem_positions,
            )

            all_compressed_embeddings.append(compressed_emb)

        # Concatenate compressed embeddings
        compressed_embeddings = torch.cat(all_compressed_embeddings, dim=1)

        # Generate
        return self.generator.generate(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            compressed_embeddings=compressed_embeddings,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )

    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            # Convert config to dict
            config_dict = {
                "compressor_type": self.config.compressor_type,
                "num_compressor_layers": self.config.num_compressor_layers,
                "modernbert_model_name": self.config.modernbert_model_name,
                "num_memory_tokens": self.config.num_memory_tokens,
                "compression_rate": self.config.compression_rate,
                "generator_model_name": self.config.generator_model_name,
                "enable_reranking": self.config.enable_reranking,
            }
            json.dump(config_dict, f, indent=2)

        # Save compressor
        compressor_path = os.path.join(save_directory, "compressor")
        os.makedirs(compressor_path, exist_ok=True)
        torch.save(self.compressor.state_dict(), os.path.join(compressor_path, "pytorch_model.bin"))

        # Save generator (LoRA adapters)
        generator_path = os.path.join(save_directory, "generator")
        if hasattr(self.generator.model, 'save_pretrained'):
            self.generator.model.save_pretrained(generator_path)
        else:
            torch.save(self.generator.state_dict(), os.path.join(generator_path, "pytorch_model.bin"))

        logger.info(f"Model saved to {save_directory}")
