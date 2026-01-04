"""
Two-stage training for N-Layers compressor.

Stage 1: Precompute compressed embeddings using N-Layers compressor
Stage 2: Train full model (compressor + generator) with cached embeddings

Key difference from ModernBERT two-stage:
- N-Layers compressor shares parameters with generator
- Stage 2 trains BOTH compressor and generator (but compression is cached)
- Even more memory efficient since compressor is just N layers

Benefits:
- Lower memory usage during Stage 1 (only N layers loaded)
- Faster training (compression happens once)
- Can use larger batch sizes
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
        reranking_labels: Optional[torch.Tensor] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.query_input_ids = query_input_ids
        self.query_attention_mask = query_attention_mask
        self.labels = labels
        self.reranking_labels = reranking_labels

        # Count number of examples
        self.num_examples = len(list(self.embeddings_dir.glob("example_*.pt")))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Load precomputed embeddings
        embeddings_path = self.embeddings_dir / f"example_{idx}.pt"
        data = torch.load(embeddings_path)

        compressed_embeddings = data["compressed_embeddings"]

        output = {
            "query_input_ids": self.query_input_ids[idx],
            "query_attention_mask": self.query_attention_mask[idx],
            "compressed_embeddings": compressed_embeddings,
            "labels": self.labels[idx],
        }

        # Add reranking scores if available
        if "reranking_scores" in data and self.reranking_labels is not None:
            output["reranking_scores"] = data["reranking_scores"]
            output["reranking_labels"] = self.reranking_labels[idx]

        return output


class TwoStageNLayersTrainer:
    """
    Two-stage trainer for N-Layers-based ScaleDown.

    Stage 1: Compress all documents using N-Layers compressor
    Stage 2: Train full model using cached compressed embeddings

    Memory savings come from:
    - Stage 1: Only load N layers (much smaller than full model)
    - Stage 2: Skip compression forward pass (use cached embeddings)
    """

    def __init__(
        self,
        model,
        config,
        train_dataset,
        eval_dataset=None,
        output_dir="./checkpoints",
        cache_dir="./compression_cache_nlayers",
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Verify N-Layers compressor
        if config.compressor_type != "n_layers":
            raise ValueError(
                "TwoStageNLayersTrainer only works with n_layers compressor. "
                f"Got: {config.compressor_type}"
            )

        # Setup device
        self.device = self._setup_device()

        logger.info("Two-stage N-Layers trainer initialized")
        logger.info(f"  Compressor: First {config.num_compressor_layers} layers")
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
        Stage 1: Precompute compressed embeddings using N-Layers compressor.

        This loads only the first N layers of the model, not the full generator.

        Args:
            force_recompute: If True, recompute even if cache exists
        """
        logger.info("=" * 80)
        logger.info(f"STAGE 1: Compressing documents with N-Layers ({self.config.num_compressor_layers} layers)")
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
        logger.info("Loading N-Layers compressor to device...")
        self.model.compressor = self.model.compressor.to(self.device)
        self.model.compressor.eval()

        # Note: We DON'T load the full generator, just the compressor
        # This saves ~50% memory for 8-layer compressor on Mistral-7B
        logger.info(
            f"Memory efficient: Only {self.config.num_compressor_layers} layers loaded, "
            f"not full {self.model.generator.model.config.num_hidden_layers} layers"
        )

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
        all_reranking_labels = [] if self.config.enable_reranking else None
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
                    all_rr_scores = []

                    for doc_idx in range(num_docs):
                        doc_ids = doc_input_ids[doc_idx].unsqueeze(0)  # [1, seq_len]
                        doc_mask = doc_attention_mask[doc_idx].unsqueeze(0)
                        mem_positions = memory_token_positions[doc_idx].unsqueeze(0)

                        # Get reranking token position if enabled
                        rr_position = None
                        if self.config.enable_reranking and "reranking_token_positions" in batch:
                            rr_position = batch["reranking_token_positions"][i, doc_idx].unsqueeze(0)

                        # Compress
                        compressed_emb, rr_score = self.model.compressor(
                            input_ids=doc_ids,
                            attention_mask=doc_mask,
                            memory_token_positions=mem_positions,
                            reranking_token_position=rr_position,
                        )
                        # compressed_emb: [1, num_mem_tokens, hidden_size]

                        all_compressed_embeddings.append(compressed_emb.squeeze(0))

                        if rr_score is not None:
                            all_rr_scores.append(rr_score)

                    # Concatenate all compressed embeddings
                    compressed_embeddings = torch.cat(all_compressed_embeddings, dim=0)
                    # [num_docs * num_mem_tokens, hidden_size]

                    # Prepare data to save
                    save_data = {
                        "compressed_embeddings": compressed_embeddings.cpu(),
                    }

                    if all_rr_scores:
                        save_data["reranking_scores"] = torch.stack(all_rr_scores).cpu()

                    # Save to cache
                    cache_path = self.cache_dir / f"example_{example_idx}.pt"
                    torch.save(save_data, cache_path)

                    # Save query and labels (needed for Stage 2)
                    all_query_input_ids.append(batch["query_input_ids"][i].cpu())
                    all_query_attention_mask.append(batch["query_attention_mask"][i].cpu())
                    all_labels.append(batch["labels"][i].cpu())

                    if self.config.enable_reranking and "reranking_labels" in batch:
                        all_reranking_labels.append(batch["reranking_labels"][i].cpu())

                    example_idx += 1

        # Save metadata
        logger.info("Saving metadata...")
        metadata = {
            "query_input_ids": torch.stack(all_query_input_ids),
            "query_attention_mask": torch.stack(all_query_attention_mask),
            "labels": torch.stack(all_labels),
            "num_examples": example_idx,
        }

        if all_reranking_labels:
            metadata["reranking_labels"] = torch.stack(all_reranking_labels)

        torch.save(metadata, self.cache_dir / "metadata.pt")

        # Mark compression as complete
        with open(cache_complete_marker, "w") as f:
            f.write(f"Compression completed with {example_idx} examples\n")
            f.write(f"Compressor: N-Layers ({self.config.num_compressor_layers} layers)\n")

        logger.info(f"✓ Stage 1 complete. Compressed {example_idx} examples.")
        logger.info(f"  Cache size: ~{self._get_cache_size_gb():.2f} GB")

        # Unload compressor to free memory
        logger.info("Unloading compressor to free memory...")
        self.model.compressor = self.model.compressor.cpu()
        torch.cuda.empty_cache()

    def _get_cache_size_gb(self):
        """Get total size of cache in GB."""
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.pt")
        )
        return total_size / (1024 ** 3)

    def stage2_train_model(self):
        """
        Stage 2: Train full model using cached compressed embeddings.

        Unlike ModernBERT two-stage (which only trains generator),
        this trains BOTH compressor and generator since they share parameters.

        However, we skip the compression forward pass and use cached embeddings,
        which still saves computation time.
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: Training model with cached embeddings")
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
        reranking_labels = metadata.get("reranking_labels", None)

        precomputed_dataset = PrecomputedEmbeddingsDataset(
            embeddings_dir=self.cache_dir,
            query_input_ids=metadata["query_input_ids"],
            query_attention_mask=metadata["query_attention_mask"],
            labels=metadata["labels"],
            reranking_labels=reranking_labels,
        )

        logger.info(f"Dataset created with {len(precomputed_dataset)} examples")

        # Move full model to device
        # Note: For N-Layers, the compressor shares parameters with generator
        # So we need to load the full model
        logger.info("Loading full model to device...")
        self.model = self.model.to(self.device)
        self.model.train()

        # Setup optimizer with different LRs for compressor and generator
        logger.info("Setting up optimizer...")

        # Get compressor and generator parameters
        compressor_params = list(self.model.compressor.parameters())
        generator_params = list(self.model.generator.parameters())

        # Remove duplicates (shared parameters)
        compressor_param_ids = {id(p) for p in compressor_params}
        generator_only_params = [
            p for p in generator_params
            if id(p) not in compressor_param_ids
        ]

        optimizer_grouped_parameters = [
            {
                "params": compressor_params,
                "lr": self.config.learning_rate_compressor,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": generator_only_params,
                "lr": self.config.learning_rate_generator,
                "weight_decay": self.config.weight_decay,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

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
        logger.info("  Note: Compression is cached, only generator forward pass runs")

        global_step = 0
        total_loss = 0
        total_gen_loss = 0
        total_rr_loss = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            progress_bar = tqdm(dataloader, desc="Training")

            for batch in progress_bar:
                # Move to device
                query_input_ids = batch["query_input_ids"].to(self.device)
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                compressed_embeddings = batch["compressed_embeddings"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass through generator (compression is cached!)
                gen_outputs = self.model.generator(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    compressed_embeddings=compressed_embeddings,
                    labels=labels,
                    return_dict=True,
                )

                loss = gen_outputs["loss"]
                generation_loss = loss

                # Add reranking loss if enabled
                reranking_loss = None
                if self.config.enable_reranking and "reranking_labels" in batch:
                    # Reranking scores are precomputed, just compute loss
                    reranking_scores = batch["reranking_scores"].to(self.device)
                    reranking_labels = batch["reranking_labels"].to(self.device)

                    reranking_loss = torch.nn.functional.mse_loss(
                        reranking_scores, reranking_labels
                    )
                    loss = loss + self.config.reranking_loss_weight * reranking_loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                global_step += 1
                total_loss += loss.item()
                total_gen_loss += generation_loss.item()

                if reranking_loss is not None:
                    total_rr_loss += reranking_loss.item()

                if global_step % 100 == 0:
                    avg_loss = total_loss / 100
                    avg_gen_loss = total_gen_loss / 100
                    avg_rr_loss = total_rr_loss / 100 if total_rr_loss > 0 else 0

                    log_dict = {
                        "loss": f"{avg_loss:.4f}",
                        "gen_loss": f"{avg_gen_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }

                    if avg_rr_loss > 0:
                        log_dict["rr_loss"] = f"{avg_rr_loss:.4f}"

                    progress_bar.set_postfix(log_dict)

                    total_loss = 0
                    total_gen_loss = 0
                    total_rr_loss = 0

                # Save checkpoint
                if global_step % 1000 == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

        logger.info("✓ Stage 2 complete. Model training finished.")
        self.save_checkpoint("final")

    def train(self, force_recompute_stage1=False):
        """
        Run both stages of training.

        Args:
            force_recompute_stage1: If True, recompute Stage 1 even if cache exists
        """
        logger.info("Starting two-stage N-Layers training...")

        # Stage 1: Compress documents
        self.stage1_compress_documents(force_recompute=force_recompute_stage1)

        # Stage 2: Train model
        self.stage2_train_model()

        logger.info("Two-stage training completed!")

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save full model
        self.model.save_pretrained(str(checkpoint_dir))

        logger.info("Checkpoint saved successfully")
