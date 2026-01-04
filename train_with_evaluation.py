"""
Train ScaleDown with before/after evaluation and performance plotting.

This script:
1. Evaluates model BEFORE training (baseline)
2. Trains the model
3. Evaluates model AFTER training
4. Plots comparison (metrics + training curves)
5. Saves comprehensive report

Usage:
    python train_with_evaluation.py --train_data small_real_dataset.json
"""

# Add current directory to path (no package installation needed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import torch

from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset, collate_fn
from scaledown.training import ScaleDownTrainer
from scaledown.evaluation import (
    ScaleDownEvaluator,
    compute_rag_metrics,
    plot_before_after_comparison,
    plot_training_curves,
    plot_inference_speed_comparison,
    save_metrics_report,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_data(train_file: str, eval_file: str = None):
    """Load training and evaluation data."""
    print(f"\nLoading training data from {train_file}...")
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    print(f"✓ Loaded {len(train_data)} training examples")

    eval_data = None
    if eval_file:
        print(f"\nLoading evaluation data from {eval_file}...")
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        print(f"✓ Loaded {len(eval_data)} evaluation examples")
    elif train_file.endswith('.json'):
        # Look for default eval file
        eval_file_auto = train_file.replace('.json', '_eval.json')
        if Path(eval_file_auto).exists():
            print(f"\nFound evaluation file: {eval_file_auto}")
            with open(eval_file_auto, 'r') as f:
                eval_data = json.load(f)
            print(f"✓ Loaded {len(eval_data)} evaluation examples")

    return train_data, eval_data


def evaluate_before_training(model, tokenizer, eval_dataset, config, output_dir):
    """Evaluate model before training."""
    print("\n" + "=" * 80)
    print("EVALUATING BEFORE TRAINING (Baseline)")
    print("=" * 80)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    evaluator = ScaleDownEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=config.device_type if config.device_type == "cuda" else "cpu",
    )

    # Evaluate quality
    metrics, predictions, inference_times = evaluator.evaluate(
        eval_dataloader,
        max_samples=100,  # Evaluate on first 100 samples
    )

    # Benchmark speed
    speed_stats = evaluator.benchmark_speed(
        eval_dataloader,
        num_trials=50,
    )

    # Save results
    results = {
        "metrics": metrics,
        "speed_stats": speed_stats,
        "sample_predictions": predictions[:5],  # First 5 predictions
    }

    with open(f"{output_dir}/before_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Before-training results saved to {output_dir}/before_training_results.json")

    return metrics, inference_times, speed_stats


def evaluate_after_training(model, tokenizer, eval_dataset, config, output_dir):
    """Evaluate model after training."""
    print("\n" + "=" * 80)
    print("EVALUATING AFTER TRAINING")
    print("=" * 80)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    evaluator = ScaleDownEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=config.device_type if config.device_type == "cuda" else "cpu",
    )

    # Evaluate quality
    metrics, predictions, inference_times = evaluator.evaluate(
        eval_dataloader,
        max_samples=100,
    )

    # Benchmark speed
    speed_stats = evaluator.benchmark_speed(
        eval_dataloader,
        num_trials=50,
    )

    # Save results
    results = {
        "metrics": metrics,
        "speed_stats": speed_stats,
        "sample_predictions": predictions[:5],
    }

    with open(f"{output_dir}/after_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ After-training results saved to {output_dir}/after_training_results.json")

    return metrics, inference_times, speed_stats


def main():
    """Main training script with evaluation."""
    parser = argparse.ArgumentParser(
        description="Train ScaleDown with before/after evaluation"
    )

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Training data JSON file")
    parser.add_argument("--eval_data", type=str, default=None, help="Evaluation data JSON file")

    # Model
    parser.add_argument("--compressor_type", type=str, default="modernbert",
                       choices=["n_layers", "modernbert"], help="Compressor type")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of compressor layers (n_layers only)")
    parser.add_argument("--num_memory_tokens", type=int, default=8, help="Number of memory tokens")
    parser.add_argument("--compression_rate", type=int, default=16, help="Compression rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="./training_with_eval",
                       help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")

    # Device
    parser.add_argument("--device", type=str, default="gpu",
                       choices=["gpu", "cpu", "trainium"], help="Device type")

    args = parser.parse_args()

    print("=" * 80)
    print("SCALEDOWN TRAINING WITH EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Train data: {args.train_data}")
    print(f"  Eval data: {args.eval_data or 'auto-detect'}")
    print(f"  Compressor: {args.compressor_type}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data, eval_data = load_data(args.train_data, args.eval_data)

    if eval_data is None:
        print("\n⚠ No evaluation data found. Splitting training data 90/10...")
        split_idx = int(len(train_data) * 0.9)
        eval_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        print(f"✓ Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Create configuration
    config = ScaleDownConfig(
        compressor_type=args.compressor_type,
        num_compressor_layers=args.num_layers if args.compressor_type == "n_layers" else None,
        num_memory_tokens=args.num_memory_tokens,
        compression_rate=args.compression_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate_generator=args.learning_rate,
        logging_steps=args.logging_steps,
        device_type=args.device,
    )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ScaleDownDataset(train_data, config)
    eval_dataset = ScaleDownDataset(eval_data, config)
    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Eval dataset: {len(eval_dataset)} examples")

    # Create model
    print("\nInitializing model...")
    model = ScaleDownModel(config)
    print(f"✓ Model initialized: {args.compressor_type} compressor")

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.generator_model_name)

    # Evaluate BEFORE training
    before_metrics, before_times, before_speed = evaluate_before_training(
        model, tokenizer, eval_dataset, config, output_dir
    )

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    trainer = ScaleDownTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir / "checkpoints"),
    )

    # Train and capture losses
    print("\nStarting training...")
    trainer.train()

    # Get training history
    train_losses = trainer.train_losses if hasattr(trainer, 'train_losses') else []

    # Evaluate AFTER training
    after_metrics, after_times, after_speed = evaluate_after_training(
        model, tokenizer, eval_dataset, config, output_dir
    )

    # Plot comparisons
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Plot 1: Before/After metrics comparison
    plot_before_after_comparison(
        before_metrics,
        after_metrics,
        output_path=str(output_dir / "before_after_comparison.png"),
    )

    # Plot 2: Training curves
    if train_losses:
        plot_training_curves(
            train_losses,
            eval_metrics=None,  # Could add if we track eval during training
            output_path=str(output_dir / "training_curves.png"),
            title=f"ScaleDown Training ({args.compressor_type})",
        )

    # Plot 3: Speed comparison
    plot_inference_speed_comparison(
        before_times,
        after_times,
        output_path=str(output_dir / "speed_comparison.png"),
    )

    # Save comprehensive report
    save_metrics_report(
        before_metrics,
        after_metrics,
        speed_stats={
            "before_mean_ms": before_speed["mean_ms"],
            "after_mean_ms": after_speed["mean_ms"],
            "speedup": before_speed["mean_ms"] / after_speed["mean_ms"],
        },
        output_path=str(output_dir / "metrics_report.json"),
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\n  Plots:")
    print(f"    - before_after_comparison.png")
    print(f"    - training_curves.png")
    print(f"    - speed_comparison.png")
    print(f"\n  Data:")
    print(f"    - metrics_report.json")
    print(f"    - before_training_results.json")
    print(f"    - after_training_results.json")
    print(f"\n  Model:")
    print(f"    - checkpoints/final/")
    print("=" * 80)


if __name__ == "__main__":
    main()
