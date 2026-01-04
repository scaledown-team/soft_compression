"""
Two-stage training for ModernBERT compressor.

Stage 1: Precompute compressed embeddings using ModernBERT
Stage 2: Train generator using cached embeddings

This approach:
- Reduces memory usage (only one model loaded at a time)
- Speeds up training (compression happens once)
- Enables larger batch sizes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict, List
import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingsDataset(Dataset):
    """Dataset that loads precomputed compressed embeddings."""

    def __init__(
        self,
        embeddings_dir: str,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.query_input_ids = query_input_ids
        self.query_attention_mask = query_attention_mask
        self.labels = labels

        # Count number of examples
        self.num_examples = len(list(self.embeddings_dir.glob("example_*.pt")))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Load precomputed embeddings
        embeddings_path = self.embeddings_dir / f"example_{idx}.pt"
        compressed_embeddings = torch.load(embeddings_path)

        return {
            "query_input_ids": self.query_input_ids[idx],
            "query_attention_mask": self.query_attention_mask[idx],
            "compressed_embeddings": compressed_embeddings,
            "labels": self.labels[idx],
        }


class TwoStageModernBERTTrainer:
    """
    Two-stage trainer for ModernBERT-based ScaleDown.

    Stage 1: Compress all documents using ModernBERT
    Stage 2: Train generator using cached compressed embeddings
    """

    def __init__(
        self,
        model,
        config,
        train_dataset,
        eval_dataset=None,
        output_dir="./checkpoints",
        cache_dir="./compression_cache",
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Verify ModernBERT compressor
        if config.compressor_type != "modernbert":
            raise ValueError(
                "TwoStageModernBERTTrainer only works with modernbert compressor. "
                f"Got: {config.compressor_type}"
            )

        # Setup device
        self.device = self._setup_device()

        logger.info("Two-stage ModernBERT trainer initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Output directory: {self.output_dir}")

    def _setup_device(self):
        """Setup device (GPU or CPU)."""
        if self.config.device_type == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")
            return device
        else:
            return torch.device("cpu")

    def stage1_compress_documents(self, force_recompute=False):
        """
        Stage 1: Precompute compressed embeddings using ModernBERT.

        Args:
            force_recompute: If True, recompute even if cache exists
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: Compressing documents with ModernBERT")
        logger.info("=" * 80)

        # Check if cache already exists
        cache_complete_marker = self.cache_dir / "compression_complete.txt"
        if cache_complete_marker.exists() and not force_recompute:
            logger.info("Compressed embeddings cache already exists. Skipping Stage 1.")
            logger.info(f"  To recompute, delete: {cache_complete_marker}")
            return

        # Clear cache directory if recomputing
        if force_recompute:
            logger.info("Force recompute enabled. Clearing cache...")
            for f in self.cache_dir.glob("example_*.pt"):
                f.unlink()

        # Move compressor to device
        logger.info("Loading ModernBERT compressor to device...")
        self.model.compressor = self.model.compressor.to(self.device)
        self.model.compressor.eval()

        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Keep order for caching
            num_workers=4,
            pin_memory=(self.config.device_type == "gpu"),
        )

        # Compress all documents
        logger.info(f"Compressing {len(self.train_dataset)} examples...")

        all_query_input_ids = []
        all_query_attention_mask = []
        all_labels = []
        example_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Compressing"):
                batch_size = batch["query_input_ids"].shape[0]

                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Compress each example in batch
                for i in range(batch_size):
                    # Get single example
                    doc_input_ids = batch["doc_input_ids"][i]  # [num_docs, seq_len]
                    doc_attention_mask = batch["doc_attention_mask"][i]
                    memory_token_positions = batch["memory_token_positions"][i]

                    num_docs = doc_input_ids.shape[0]

                    # Compress all documents for this example
                    all_compressed_embeddings = []

                    for doc_idx in range(num_docs):
                        doc_ids = doc_input_ids[doc_idx].unsqueeze(0)  # [1, seq_len]
                        doc_mask = doc_attention_mask[doc_idx].unsqueeze(0)
                        mem_positions = memory_token_positions[doc_idx].unsqueeze(0)

                        # Compress
                        compressed_emb, _ = self.model.compressor(
                            input_ids=doc_ids,
                            attention_mask=doc_mask,
                            memory_token_positions=mem_positions,
                        )
                        # compressed_emb: [1, num_mem_tokens, hidden_size]

                        all_compressed_embeddings.append(compressed_emb.squeeze(0))

                    # Concatenate all compressed embeddings
                    compressed_embeddings = torch.cat(all_compressed_embeddings, dim=0)
                    # [num_docs * num_mem_tokens, hidden_size]

                    # Save to cache (on CPU to save GPU memory)
                    cache_path = self.cache_dir / f"example_{example_idx}.pt"
                    torch.save(compressed_embeddings.cpu(), cache_path)

                    # Save query and labels (needed for Stage 2)
                    all_query_input_ids.append(batch["query_input_ids"][i].cpu())
                    all_query_attention_mask.append(batch["query_attention_mask"][i].cpu())
                    all_labels.append(batch["labels"][i].cpu())

                    example_idx += 1

        # Save metadata
        logger.info("Saving metadata...")
        metadata = {
            "query_input_ids": torch.stack(all_query_input_ids),
            "query_attention_mask": torch.stack(all_query_attention_mask),
            "labels": torch.stack(all_labels),
            "num_examples": example_idx,
        }
        torch.save(metadata, self.cache_dir / "metadata.pt")

        # Mark compression as complete
        with open(cache_complete_marker, "w") as f:
            f.write(f"Compression completed with {example_idx} examples\n")

        logger.info(f"✓ Stage 1 complete. Compressed {example_idx} examples.")
        logger.info(f"  Cache size: ~{self._get_cache_size_gb():.2f} GB")

        # Unload compressor to free memory
        logger.info("Unloading ModernBERT compressor to free memory...")
        self.model.compressor = self.model.compressor.cpu()
        torch.cuda.empty_cache()

    def _get_cache_size_gb(self):
        """Get total size of cache in GB."""
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.pt")
        )
        return total_size / (1024 ** 3)

    def stage2_train_generator(self):
        """
        Stage 2: Train generator using cached compressed embeddings.

        This only loads the generator (+ LoRA) into memory, not the compressor.
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: Training generator with cached embeddings")
        logger.info("=" * 80)

        # Check if cache exists
        cache_complete_marker = self.cache_dir / "compression_complete.txt"
        if not cache_complete_marker.exists():
            raise RuntimeError(
                "Compressed embeddings cache not found. "
                "Run stage1_compress_documents() first."
            )

        # Load metadata
        logger.info("Loading metadata...")
        metadata = torch.load(self.cache_dir / "metadata.pt")

        # Create dataset with precomputed embeddings
        logger.info("Creating dataset with precomputed embeddings...")
        precomputed_dataset = PrecomputedEmbeddingsDataset(
            embeddings_dir=self.cache_dir,
            query_input_ids=metadata["query_input_ids"],
            query_attention_mask=metadata["query_attention_mask"],
            labels=metadata["labels"],
        )

        logger.info(f"Dataset created with {len(precomputed_dataset)} examples")

        # Move only generator to device
        logger.info("Loading generator to device...")
        self.model.generator = self.model.generator.to(self.device)
        self.model.generator.train()

        # Setup optimizer (only for generator parameters)
        logger.info("Setting up optimizer for generator...")
        optimizer = AdamW(
            self.model.generator.parameters(),
            lr=self.config.learning_rate_generator,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        num_training_steps = len(precomputed_dataset) // self.config.batch_size * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        # Create dataloader
        dataloader = DataLoader(
            precomputed_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Embeddings are already on disk
            pin_memory=(self.config.device_type == "gpu"),
        )

        # Training loop
        logger.info("Starting training...")
        global_step = 0
        total_loss = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            progress_bar = tqdm(dataloader, desc="Training")

            for batch in progress_bar:
                # Move to device
                query_input_ids = batch["query_input_ids"].to(self.device)
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                compressed_embeddings = batch["compressed_embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass (generator only)
                outputs = self.model.generator(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    compressed_embeddings=compressed_embeddings,
                    labels=labels,
                    return_dict=True,
                )

                loss = outputs["loss"]

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.generator.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                global_step += 1
                total_loss += loss.item()

                if global_step % 100 == 0:
                    avg_loss = total_loss / 100
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })
                    total_loss = 0

                # Save checkpoint
                if global_step % 1000 == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

        logger.info("✓ Stage 2 complete. Generator training finished.")
        self.save_checkpoint("final")

    def train(self, force_recompute_stage1=False):
        """
        Run both stages of training.

        Args:
            force_recompute_stage1: If True, recompute Stage 1 even if cache exists
        """
        logger.info("Starting two-stage training...")

        # Stage 1: Compress documents
        self.stage1_compress_documents(force_recompute=force_recompute_stage1)

        # Stage 2: Train generator
        self.stage2_train_generator()

        logger.info("Two-stage training completed!")

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save full model (both compressor and generator)
        self.model.save_pretrained(str(checkpoint_dir))

        logger.info("Checkpoint saved successfully")
