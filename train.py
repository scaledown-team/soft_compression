"""
Main training script for ScaleDown.

Example usage:
    # Train with N-Layers compressor on GPU
    python train.py --compressor_type n_layers --num_layers 8 --device gpu

    # Train with ModernBERT compressor on Trainium
    python train.py --compressor_type modernbert --device trainium

    # Enable reranking
    python train.py --compressor_type n_layers --enable_reranking
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import set_seed

from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data.dataset import ScaleDownDataset
from scaledown.training.trainer import ScaleDownTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ScaleDown model")

    # Model configuration
    parser.add_argument(
        "--compressor_type",
        type=str,
        default="n_layers",
        choices=["n_layers", "modernbert"],
        help="Type of compressor to use"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of layers for N-Layers compressor (5, 8, or 10)"
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
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON file)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (JSON file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
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
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate_generator",
        type=float,
        default=1e-4,
        help="Learning rate for generator"
    )
    parser.add_argument(
        "--learning_rate_compressor",
        type=float,
        default=None,
        help="Learning rate for compressor (default: auto based on type)"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "trainium"],
        help="Device type for training"
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
    logger.info("ScaleDown Training")
    logger.info("=" * 80)
    logger.info(f"Compressor type: {args.compressor_type}")
    logger.info(f"Generator model: {args.generator_model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Reranking: {args.enable_reranking}")
    logger.info("=" * 80)

    # Create configuration
    config = ScaleDownConfig(
        compressor_type=args.compressor_type,
        num_compressor_layers=args.num_layers,
        generator_model_name=args.generator_model,
        enable_reranking=args.enable_reranking,
        device_type=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate_generator=args.learning_rate_generator,
    )

    # Override compressor learning rate if specified
    if args.learning_rate_compressor is not None:
        if args.compressor_type == "n_layers":
            config.learning_rate_compressor_nlayers = args.learning_rate_compressor
        elif args.compressor_type == "modernbert":
            config.learning_rate_compressor_modernbert = args.learning_rate_compressor

    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    logger.info(f"Loaded {len(train_data)} training examples")

    eval_data = None
    if args.eval_data:
        logger.info(f"Loading evaluation data from {args.eval_data}")
        eval_data = load_data(args.eval_data)
        logger.info(f"Loaded {len(eval_data)} evaluation examples")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ScaleDownDataset(train_data, config)

    eval_dataset = None
    if eval_data:
        eval_dataset = ScaleDownDataset(eval_data, config)

    # Create model
    logger.info("Initializing model...")
    model = ScaleDownModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = ScaleDownTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed successfully!")
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
