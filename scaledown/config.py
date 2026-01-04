"""Configuration classes for ScaleDown model."""

from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class ScaleDownConfig:
    """Configuration for ScaleDown model.

    Based on OSCAR architecture with two compressor options:
    1. N-Layers: First N layers of generator (faithful to paper)
    2. ModernBERT: ModernBERT-base encoder (novel variant)
    """

    # Compressor configuration
    compressor_type: Literal["n_layers", "modernbert"] = "n_layers"

    # For N-Layers compressor
    num_compressor_layers: int = 8  # Paper tested: 5, 8, 10

    # For ModernBERT compressor
    modernbert_model_name: str = "answerdotai/ModernBERT-base"

    # Compression parameters
    num_memory_tokens: int = 8
    compression_rate: int = 16  # Paper used 16x (8 mem tokens for 128 token docs)

    # Generator configuration
    generator_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # LoRA configuration (same as paper)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )

    # Reranking configuration
    enable_reranking: bool = False
    reranking_loss_weight: float = 0.05  # λ in paper

    # Training configuration (from paper)
    max_seq_length: int = 128  # Document chunk size
    num_documents_train: int = 5  # Paper: trained with 5 docs
    num_documents_eval: int = 10  # Paper: evaluated with 10 docs
    max_answer_length: int = 128

    # Training hyperparameters (from paper Table 7)
    batch_size: int = 128
    learning_rate_generator: float = 1e-4
    learning_rate_compressor_llama: float = 1e-4
    learning_rate_compressor_nlayers: float = 5e-5
    learning_rate_compressor_modernbert: float = 1e-4  # Same as llama
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Optimizer configuration
    optimizer: str = "adamw"
    lr_scheduler: str = "linear"

    # Device configuration
    device_type: Literal["gpu", "trainium"] = "gpu"

    # Mixed precision
    use_bf16: bool = True  # BF16 for modern GPUs/Trainium
    use_fp16: bool = False

    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True

    # Hidden dimensions (will be auto-detected from models)
    compressor_hidden_size: Optional[int] = None
    generator_hidden_size: Optional[int] = None

    # Projection layer for ModernBERT (768D -> 4096D for Mistral-7B)
    projection_hidden_size: int = 4096

    def __post_init__(self):
        """Validate configuration."""
        if self.compression_rate not in [8, 16, 32, 64, 128]:
            raise ValueError(
                f"compression_rate must be one of [8, 16, 32, 64, 128], "
                f"got {self.compression_rate}"
            )

        if self.num_memory_tokens * self.compression_rate > self.max_seq_length:
            raise ValueError(
                f"num_memory_tokens ({self.num_memory_tokens}) * "
                f"compression_rate ({self.compression_rate}) must be <= "
                f"max_seq_length ({self.max_seq_length})"
            )

        if self.compressor_type == "n_layers":
            if self.num_compressor_layers < 1:
                raise ValueError(
                    f"num_compressor_layers must be >= 1, got {self.num_compressor_layers}"
                )

        # Set appropriate learning rate based on compressor type
        if self.compressor_type == "n_layers":
            self.learning_rate_compressor = self.learning_rate_compressor_nlayers
        elif self.compressor_type == "modernbert":
            self.learning_rate_compressor = self.learning_rate_compressor_modernbert

    @property
    def learning_rate_compressor(self) -> float:
        """Get learning rate for compressor based on type."""
        if self.compressor_type == "n_layers":
            return self.learning_rate_compressor_nlayers
        elif self.compressor_type == "modernbert":
            return self.learning_rate_compressor_modernbert
        else:
            raise ValueError(f"Unknown compressor type: {self.compressor_type}")

    @learning_rate_compressor.setter
    def learning_rate_compressor(self, value: float):
        """Set learning rate (used in __post_init__)."""
        pass  # Setter needed for property
