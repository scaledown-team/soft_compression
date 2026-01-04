"""
Generator module for ScaleDown.

Implements the LLM generator with LoRA fine-tuning, following OSCAR paper.
Supports both GPU and AWS Trainium via device abstraction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ScaleDownGenerator(nn.Module):
    """
    Generator LLM with LoRA adapters.

    Architecture (from OSCAR paper):
        - Base model: Mistral-7B-Instruct (or other causal LM)
        - Fine-tuning: LoRA adapters (r=16, alpha=32, dropout=0.1)
        - Training: Sentence-level distillation from teacher LLM

    Input format:
        [Query tokens] [Compressed doc 1 embeddings] [Compressed doc 2 embeddings] ...

    The compressed embeddings from the compressor replace the original document tokens.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config

        # Device setup (GPU or Trainium)
        if device is None:
            device = self._setup_device(config.device_type)
        self.device = device

        logger.info(f"Loading generator model: {config.generator_model_name}")
        logger.info(f"Device: {self.device}")

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if config.use_bf16 else torch.float32,
        }

        # For Trainium, load in float32 and XLA will handle optimization
        if config.device_type == "trainium":
            model_kwargs["torch_dtype"] = torch.float32
            logger.info("Loading model in FP32 for Trainium (XLA will optimize)")

        # For GPU, load on CPU first to avoid OOM during initialization
        # The trainer will move to GPU after full model is created
        if config.device_type == "gpu":
            model_kwargs["device_map"] = None  # Load on CPU first
            model_kwargs["low_cpu_mem_usage"] = True
            logger.info("Loading generator on CPU first (trainer will move to GPU)")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.generator_model_name,
            **model_kwargs
        )

        # Apply LoRA
        if config.use_lora:
            logger.info("Applying LoRA adapters")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Get hidden size
        self.hidden_size = self.model.config.hidden_size

        # Don't move to device here - let the trainer handle it
        # This avoids loading multiple large models on GPU simultaneously
        logger.info("Generator initialized on CPU (trainer will handle device placement)")

    def _setup_device(self, device_type: str):
        """
        Setup device based on configuration.

        For GPU: Use CUDA if available
        For Trainium: Use XLA device via torch_neuronx
        """
        if device_type == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")
            return device

        elif device_type == "trainium":
            try:
                import torch_neuronx  # noqa: F401
                device = torch.device("xla")
                logger.info("Using AWS Trainium with XLA device")
                return device
            except ImportError:
                raise ImportError(
                    "torch_neuronx not found. Please install AWS Neuron SDK:\n"
                    "pip install torch-neuronx neuronx-cc --extra-index-url "
                    "https://pip.repos.neuron.amazonaws.com"
                )
        else:
            raise ValueError(f"Unknown device_type: {device_type}")

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        compressed_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with compressed document embeddings.

        Args:
            query_input_ids: [batch_size, query_len] - Query token IDs
            query_attention_mask: [batch_size, query_len] - Query attention mask
            compressed_embeddings: [batch_size, num_docs * num_mem_tokens, hidden_size]
                - Compressed document embeddings from compressor
            labels: [batch_size, seq_len] - Labels for generation (teacher answers)
            return_dict: Whether to return dict

        Returns:
            Dictionary with 'loss' and 'logits'
        """
        batch_size = query_input_ids.shape[0]

        # Get query embeddings
        # Handle PEFT-wrapped models
        if hasattr(self.model, 'base_model'):
            embed_tokens = self.model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.model.model.embed_tokens

        query_embeddings = embed_tokens(query_input_ids)
        # [batch_size, query_len, hidden_size]

        # Concatenate query embeddings with compressed document embeddings
        inputs_embeds = torch.cat([query_embeddings, compressed_embeddings], dim=1)
        # [batch_size, query_len + num_docs * num_mem_tokens, hidden_size]

        # Create attention mask
        compressed_attention_mask = torch.ones(
            batch_size,
            compressed_embeddings.shape[1],
            dtype=query_attention_mask.dtype,
            device=query_attention_mask.device
        )
        attention_mask = torch.cat([query_attention_mask, compressed_attention_mask], dim=1)

        # Forward pass through model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        if return_dict:
            return {
                "loss": outputs.loss if labels is not None else None,
                "logits": outputs.logits,
            }
        else:
            return outputs.loss if labels is not None else outputs.logits

    def generate(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        compressed_embeddings: torch.Tensor,
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate answers given query and compressed documents.

        Args:
            query_input_ids: [batch_size, query_len]
            query_attention_mask: [batch_size, query_len]
            compressed_embeddings: [batch_size, num_docs * num_mem_tokens, hidden_size]
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            generated_ids: [batch_size, generated_len]
        """
        batch_size = query_input_ids.shape[0]

        # Get query embeddings
        # Handle PEFT-wrapped models
        if hasattr(self.model, 'base_model'):
            embed_tokens = self.model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.model.model.embed_tokens

        query_embeddings = embed_tokens(query_input_ids)

        # Concatenate with compressed embeddings
        inputs_embeds = torch.cat([query_embeddings, compressed_embeddings], dim=1)

        # Create attention mask
        compressed_attention_mask = torch.ones(
            batch_size,
            compressed_embeddings.shape[1],
            dtype=query_attention_mask.dtype,
            device=query_attention_mask.device
        )
        attention_mask = torch.cat([query_attention_mask, compressed_attention_mask], dim=1)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

        return outputs

    def get_hidden_size(self) -> int:
        """Get generator hidden size."""
        return self.hidden_size
