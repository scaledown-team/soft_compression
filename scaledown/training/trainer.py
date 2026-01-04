"""
Training script for ScaleDown with GPU and AWS Trainium support.

Implements the training procedure from OSCAR paper with cross-platform compatibility.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ScaleDownTrainer:
    """
    Trainer for ScaleDown model.

    Supports:
        - GPU training with mixed precision
        - AWS Trainium training via Neuron SDK
        - LoRA fine-tuning of generator
        - Full fine-tuning of compressor
        - Distillation loss from teacher LLM
        - Optional reranking loss
    """

    def __init__(
        self,
        model,
        config,
        train_dataset,
        eval_dataset=None,
        output_dir="./checkpoints",
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = self._setup_device()

        # Move model to device (if not Trainium)
        if config.device_type != "trainium":
            self.model = self.model.to(self.device)

        # Setup optimizer with different learning rates for compressor and generator
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        num_training_steps = len(train_dataset) // config.batch_size * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        # Setup mixed precision training
        self.scaler = None
        if config.device_type == "gpu" and (config.use_fp16 or config.use_bf16):
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_fp16)

        # Trainium-specific setup
        if config.device_type == "trainium":
            self._setup_trainium()

        logger.info(f"Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Training steps: {num_training_steps}")

    def _setup_device(self):
        """Setup device (GPU or Trainium)."""
        if self.config.device_type == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")
            return device

        elif self.config.device_type == "trainium":
            try:
                import torch_neuronx  # noqa: F401

                device = torch.device("xla")
                logger.info("Using AWS Trainium with XLA")
                return device
            except ImportError:
                raise ImportError(
                    "torch_neuronx not found. Install: "
                    "pip install torch-neuronx neuronx-cc --extra-index-url "
                    "https://pip.repos.neuron.amazonaws.com"
                )

    def _setup_trainium(self):
        """Setup Trainium-specific configurations."""
        try:
            import torch_xla.core.xla_model as xm

            self.xm = xm
            logger.info("Trainium XLA model loaded")
        except ImportError:
            raise ImportError("torch_xla required for Trainium training")

    def _create_optimizer(self):
        """
        Create optimizer with different learning rates for compressor and generator.

        Following OSCAR paper Table 7:
            - Generator (LoRA): 1e-4
            - N-Layers compressor: 5e-5
            - ModernBERT compressor: 1e-4
        """
        # Separate parameters
        compressor_params = list(self.model.compressor.parameters())
        generator_params = list(self.model.generator.parameters())

        optimizer_grouped_parameters = [
            {
                "params": compressor_params,
                "lr": self.config.learning_rate_compressor,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": generator_params,
                "lr": self.config.learning_rate_generator,
                "weight_decay": self.config.weight_decay,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        logger.info(f"Optimizer created:")
        logger.info(f"  Compressor LR: {self.config.learning_rate_compressor}")
        logger.info(f"  Generator LR: {self.config.learning_rate_generator}")
        logger.info(f"  Weight decay: {self.config.weight_decay}")

        return optimizer

    def train(self):
        """
        Main training loop.

        Follows OSCAR paper training procedure:
            - Batch size: 128
            - Epochs: 1
            - Gradient accumulation if needed
            - Gradient clipping: max_norm=1.0
        """
        logger.info("Starting training...")

        # Create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.config.device_type == "gpu"),
        )

        self.model.train()
        global_step = 0
        total_loss = 0
        total_gen_loss = 0
        total_rr_loss = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            progress_bar = tqdm(train_loader, desc=f"Training")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                if self.config.device_type == "gpu" and self.scaler is not None:
                    # Mixed precision training on GPU
                    with torch.cuda.amp.autocast(
                        enabled=True,
                        dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                    ):
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Trainium-specific: mark step for XLA
                if self.config.device_type == "trainium":
                    self.xm.mark_step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Logging
                global_step += 1
                total_loss += loss.item()
                total_gen_loss += outputs["generation_loss"].item()

                if outputs["reranking_loss"] is not None:
                    total_rr_loss += outputs["reranking_loss"].item()

                if global_step % 100 == 0:
                    avg_loss = total_loss / 100
                    avg_gen_loss = total_gen_loss / 100
                    avg_rr_loss = total_rr_loss / 100 if total_rr_loss > 0 else 0

                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "gen_loss": f"{avg_gen_loss:.4f}",
                            "rr_loss": f"{avg_rr_loss:.4f}"
                            if avg_rr_loss > 0
                            else "N/A",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        }
                    )

                    total_loss = 0
                    total_gen_loss = 0
                    total_rr_loss = 0

                # Save checkpoint
                if global_step % 1000 == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

            # End of epoch
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

            # Evaluation
            if self.eval_dataset is not None:
                self.evaluate()

        logger.info("Training completed!")
        self.save_checkpoint("final")

    def evaluate(self):
        """Evaluate model on eval dataset."""
        logger.info("Running evaluation...")

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        self.model.eval()
        total_loss = 0
        total_gen_loss = 0
        total_rr_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(**batch)

                total_loss += outputs["loss"].item()
                total_gen_loss += outputs["generation_loss"].item()

                if outputs["reranking_loss"] is not None:
                    total_rr_loss += outputs["reranking_loss"].item()

                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches
        avg_rr_loss = total_rr_loss / num_batches if total_rr_loss > 0 else 0

        logger.info(f"Evaluation results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Generation loss: {avg_gen_loss:.4f}")
        if avg_rr_loss > 0:
            logger.info(f"  Reranking loss: {avg_rr_loss:.4f}")

        self.model.train()

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            checkpoint_dir / "training_state.pt",
        )

        logger.info(f"Checkpoint saved successfully")
