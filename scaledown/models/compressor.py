"""
Compressor modules for ScaleDown.

Implements two variants:
1. NLayersCompressor: First N layers of generator (faithful to OSCAR paper)
2. ModernBERTCompressor: ModernBERT encoder (novel variant)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ScaleDownCompressor(nn.Module):
    """Base class for ScaleDown compressors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_memory_tokens = config.num_memory_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_token_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compress query-document pairs.

        Args:
            input_ids: [batch_size, seq_len] - contains [Query] [Document] [MEM tokens]
            attention_mask: [batch_size, seq_len]
            memory_token_positions: [batch_size, num_memory_tokens] - positions of MEM tokens

        Returns:
            compressed_embeddings: [batch_size, num_memory_tokens, hidden_size]
            reranking_scores: [batch_size] if reranking enabled, else None
        """
        raise NotImplementedError


class NLayersCompressor(ScaleDownCompressor):
    """
    N-Layers compressor: Uses first N layers of the generator.

    This is faithful to the OSCAR paper's OSCAR-N-Layers approach.
    Key advantage: No pretraining needed as hidden representations are already aligned.

    Architecture:
        Input: [Query] [Document] [MEM_1] ... [MEM_l]
        → First N layers of generator backbone
        → Extract hidden states at memory token positions
        → Output: l compressed embeddings
    """

    def __init__(self, config, generator_model):
        super().__init__(config)

        self.num_layers = config.num_compressor_layers

        # Extract first N layers from generator
        logger.info(f"Creating N-Layers compressor with {self.num_layers} layers")

        # Get model architecture
        model_config = generator_model.config

        # Handle PEFT-wrapped models (LoRA)
        # If the generator is wrapped with PEFT, unwrap it to access the base model
        if hasattr(generator_model, 'base_model'):
            # PEFT-wrapped model: access base_model.model.embed_tokens
            base_model = generator_model.base_model.model
        else:
            # Regular model: access model.embed_tokens
            base_model = generator_model

        # Create a shallow copy of the model with only first N layers
        self.embedding = base_model.model.embed_tokens

        # Extract first N layers
        if hasattr(base_model.model, 'layers'):
            # For Mistral, Llama, Qwen, etc.
            self.layers = nn.ModuleList([
                base_model.model.layers[i] for i in range(self.num_layers)
            ])
        else:
            raise ValueError(
                f"Generator model type {type(base_model)} not supported. "
                "Expected model.layers attribute."
            )

        self.norm = base_model.model.norm

        # Reranking head (optional)
        self.enable_reranking = config.enable_reranking
        if self.enable_reranking:
            self.reranking_head = nn.Linear(model_config.hidden_size, 1)
            logger.info("Reranking head enabled")

        # Store hidden size
        self.hidden_size = model_config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_token_positions: torch.Tensor,
        reranking_token_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through first N layers.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            memory_token_positions: [batch_size, num_memory_tokens]
            reranking_token_position: [batch_size] - position of RR token if reranking

        Returns:
            compressed_embeddings: [batch_size, num_memory_tokens, hidden_size]
            reranking_scores: [batch_size] if reranking enabled
        """
        batch_size = input_ids.shape[0]

        # Embedding layer
        hidden_states = self.embedding(input_ids)

        # Create causal mask (for decoder-only models)
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Combine with attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            causal_mask = causal_mask | (~attention_mask.bool())

        # Pass through first N layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=~causal_mask if causal_mask is not None else None,
            )
            hidden_states = layer_outputs[0]

        # Apply normalization
        hidden_states = self.norm(hidden_states)

        # Extract memory token embeddings
        # memory_token_positions: [batch_size, num_memory_tokens]
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, self.num_memory_tokens)

        compressed_embeddings = hidden_states[batch_indices, memory_token_positions]
        # Shape: [batch_size, num_memory_tokens, hidden_size]

        # Compute reranking scores if enabled
        reranking_scores = None
        if self.enable_reranking and reranking_token_position is not None:
            rr_embeddings = hidden_states[
                torch.arange(batch_size, device=input_ids.device),
                reranking_token_position
            ]  # [batch_size, hidden_size]
            reranking_scores = self.reranking_head(rr_embeddings).squeeze(-1)  # [batch_size]

        return compressed_embeddings, reranking_scores


class ModernBERTCompressor(ScaleDownCompressor):
    """
    ModernBERT-based compressor (novel variant).

    This is our novel contribution - using an encoder-only model for compression.

    Rationale:
        - Compression is fundamentally an encoding task
        - ModernBERT (149M params) is much smaller than Llama-1B (1.1B params)
        - Bidirectional attention may be more effective for document understanding
        - 2x faster than decoder-only models

    Architecture:
        Input: [Query] [SEP] [Document] [MEM_1] ... [MEM_l] [RR]
        → ModernBERT-base (22 layers, bidirectional)
        → Extract hidden states at memory token positions (768D)
        → Projection: FC(768 → hidden) → ReLU → FC(hidden → generator_hidden)
        → Output: l compressed embeddings (generator_hidden_size)
    """

    def __init__(self, config):
        super().__init__(config)

        logger.info(f"Loading ModernBERT model: {config.modernbert_model_name}")

        # Load ModernBERT
        self.encoder = AutoModel.from_pretrained(
            config.modernbert_model_name,
            trust_remote_code=True
        )

        encoder_config = self.encoder.config
        self.encoder_hidden_size = encoder_config.hidden_size  # 768 for base

        # Projection layers to map encoder hidden space to generator hidden space
        generator_hidden_size = config.generator_hidden_size or 4096

        self.projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, config.projection_hidden_size),
            nn.ReLU(),
            nn.Linear(config.projection_hidden_size, generator_hidden_size)
        )

        logger.info(
            f"Projection: {self.encoder_hidden_size}D → "
            f"{config.projection_hidden_size}D → {generator_hidden_size}D"
        )

        # Reranking head (optional)
        self.enable_reranking = config.enable_reranking
        if self.enable_reranking:
            self.reranking_head = nn.Linear(self.encoder_hidden_size, 1)
            logger.info("Reranking head enabled")

        self.hidden_size = generator_hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_token_positions: torch.Tensor,
        reranking_token_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ModernBERT.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            memory_token_positions: [batch_size, num_memory_tokens]
            reranking_token_position: [batch_size] - position of RR token if reranking

        Returns:
            compressed_embeddings: [batch_size, num_memory_tokens, generator_hidden_size]
            reranking_scores: [batch_size] if reranking enabled
        """
        batch_size = input_ids.shape[0]

        # Encode with ModernBERT (bidirectional attention)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # Extract memory token embeddings
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, self.num_memory_tokens)

        memory_embeddings = hidden_states[batch_indices, memory_token_positions]
        # Shape: [batch_size, num_memory_tokens, encoder_hidden_size]

        # Project to generator hidden space
        compressed_embeddings = self.projection(memory_embeddings)
        # Shape: [batch_size, num_memory_tokens, generator_hidden_size]

        # Compute reranking scores if enabled
        reranking_scores = None
        if self.enable_reranking and reranking_token_position is not None:
            rr_embeddings = hidden_states[
                torch.arange(batch_size, device=input_ids.device),
                reranking_token_position
            ]  # [batch_size, encoder_hidden_size]
            reranking_scores = self.reranking_head(rr_embeddings).squeeze(-1)  # [batch_size]

        return compressed_embeddings, reranking_scores

    def get_encoder_hidden_size(self) -> int:
        """Get ModernBERT hidden size (768 for base)."""
        return self.encoder_hidden_size
