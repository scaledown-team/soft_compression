"""
Train ScaleDown with N-Layers using two-stage approach.

Benefits:
- Stage 1: Only load N layers (50-70% less memory than full model)
- Stage 2: Skip compression forward pass (30-40% faster)
- Larger batch sizes possible

Usage:
    # Run both stages
    python train_nlayers_two_stage.py --train_data data.json --num_layers 8

    # Run only Stage 1
    python train_nlayers_two_stage.py --train_data data.json --stage1_only

    # Run only Stage 2
    python train_nlayers_two_stage.py --train_data data.json --stage2_only
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import set_seed

from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data.dataset import ScaleDownDataset
from scaledown.training.two_stage_nlayers_trainer import TwoStageNLayersTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ScaleDown with N-Layers (two-stage)"
    )

    # Data
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON file)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./compression_cache_nlayers",
        help="Directory to cache compressed embeddings"
    )

    # Model configuration
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of compressor layers (5, 8, or 10)"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Generator model name"
    )
    parser.add_argument(
        "--enable_reranking",
        action="store_true",
        help="Enable joint compression and reranking"
    )

    # Training configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_nlayers",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (for Stage 2)"
    )

    # Stage control
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        help="Only run Stage 1 (compression)"
    )
    parser.add_argument(
        "--stage2_only",
        action="store_true",
        help="Only run Stage 2 (training) - requires Stage 1 to be done"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recompute Stage 1 even if cache exists"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def load_data(data_path: str):
    """Load training data from JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info("ScaleDown N-Layers Two-Stage Training")
    logger.info("=" * 80)
    logger.info(f"Compressor: N-Layers ({args.num_layers} layers)")
    logger.info(f"Generator: {args.generator_model}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Create configuration
    config = ScaleDownConfig(
        compressor_type="n_layers",
        num_compressor_layers=args.num_layers,
        generator_model_name=args.generator_model,
        enable_reranking=args.enable_reranking,
        device_type="gpu",
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    logger.info(f"Loaded {len(train_data)} training examples")

    # Create dataset
    logger.info("Creating dataset...")
    train_dataset = ScaleDownDataset(train_data, config)

    # Create model
    logger.info("Initializing model...")
    model = ScaleDownModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compressor_params = sum(p.numel() for p in model.compressor.parameters())

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Compressor parameters: {compressor_params:,}")
    logger.info(f"Memory saving in Stage 1: ~{(1 - compressor_params/total_params)*100:.1f}%")

    # Create two-stage trainer
    logger.info("Creating two-stage trainer...")
    trainer = TwoStageNLayersTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    # Run training based on stage flags
    if args.stage1_only:
        logger.info("Running Stage 1 only (compression)...")
        trainer.stage1_compress_documents(force_recompute=args.force_recompute)

    elif args.stage2_only:
        logger.info("Running Stage 2 only (training)...")
        trainer.stage2_train_model()

    else:
        logger.info("Running both stages...")
        trainer.train(force_recompute_stage1=args.force_recompute)

    logger.info("Training completed successfully!")
    logger.info(f"Model saved to {args.output_dir}")

    # Print cache info
    if Path(args.cache_dir).exists():
        cache_size = sum(
            f.stat().st_size
            for f in Path(args.cache_dir).glob("*.pt")
        ) / (1024 ** 3)
        logger.info(f"Compression cache size: {cache_size:.2f} GB")


if __name__ == "__main__":
    main()
