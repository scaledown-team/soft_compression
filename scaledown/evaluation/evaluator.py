"""
Model evaluation for ScaleDown.

This module provides:
- Before/after training evaluation
- Inference speed benchmarking
- Comprehensive performance reporting
"""

import time
import torch
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    compute_rag_metrics,
    plot_before_after_comparison,
    plot_inference_speed_comparison,
    save_metrics_report,
)


class ScaleDownEvaluator:
    """
    Evaluator for ScaleDown models.

    Measures:
    - Generation quality (EM, F1, ROUGE)
    - Inference speed
    - Memory usage
    """

    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize evaluator.

        Args:
            model: ScaleDown model
            tokenizer: Tokenizer for the generator
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def evaluate(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 128,
    ) -> Tuple[Dict[str, float], List[str], List[float]]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            max_samples: Maximum number of samples to evaluate (None = all)
            max_new_tokens: Maximum tokens to generate per answer

        Returns:
            Tuple of (metrics, predictions, inference_times)
        """
        predictions = []
        ground_truths = []
        inference_times = []

        num_samples = 0

        print(f"\nEvaluating on {len(dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if max_samples and num_samples >= max_samples:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Time inference
                start_time = time.time()

                # Generate answers
                outputs = self.model.generate(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    max_new_tokens=max_new_tokens,
                )

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = (time.time() - start_time) * 1000  # Convert to ms

                # Decode predictions
                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    predictions.append(pred_text)

                    # Get ground truth (if available in batch)
                    if 'answer_text' in batch:
                        ground_truths.append(batch['answer_text'][i])

                    inference_times.append(inference_time / len(outputs))  # Per sample

                num_samples += len(outputs)

        # Compute metrics
        if ground_truths:
            metrics = compute_rag_metrics(predictions, ground_truths)
        else:
            metrics = {}

        print(f"\n✓ Evaluated {len(predictions)} samples")

        return metrics, predictions, inference_times

    def benchmark_speed(
        self,
        dataloader: DataLoader,
        num_trials: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            dataloader: DataLoader with evaluation data
            num_trials: Number of inference trials

        Returns:
            Speed statistics
        """
        times = []

        print(f"\nBenchmarking speed ({num_trials} trials)...")

        # Warmup
        batch = next(iter(dataloader))
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        for _ in range(3):
            with torch.no_grad():
                self.model.generate(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    max_new_tokens=64,
                )

        # Benchmark
        for _ in tqdm(range(num_trials), desc="Benchmarking"):
            start_time = time.time()

            with torch.no_grad():
                self.model.generate(
                    query_input_ids=batch['query_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    doc_input_ids=batch['doc_input_ids'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    max_new_tokens=64,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append((time.time() - start_time) * 1000)  # ms

        stats = {
            "mean_ms": float(torch.tensor(times).mean()),
            "std_ms": float(torch.tensor(times).std()),
            "min_ms": float(torch.tensor(times).min()),
            "max_ms": float(torch.tensor(times).max()),
            "throughput_qps": 1000.0 / float(torch.tensor(times).mean()),
        }

        print(f"\n✓ Speed benchmark complete")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std: {stats['std_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_qps']:.2f} queries/sec")

        return stats


def evaluate_model(
    model,
    tokenizer,
    eval_dataloader: DataLoader,
    output_dir: str = "./evaluation_results",
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.

    Args:
        model: ScaleDown model
        tokenizer: Tokenizer
        eval_dataloader: Evaluation data loader
        output_dir: Directory to save results
        max_samples: Maximum samples to evaluate
        device: Device to use

    Returns:
        Evaluation metrics
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = ScaleDownEvaluator(model, tokenizer, device)

    # Evaluate
    metrics, predictions, inference_times = evaluator.evaluate(
        eval_dataloader,
        max_samples=max_samples,
    )

    # Save results
    import json
    results = {
        "metrics": metrics,
        "predictions": predictions[:10],  # Save first 10 for inspection
        "avg_inference_time_ms": float(torch.tensor(inference_times).mean()),
    }

    with open(f"{output_dir}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/evaluation_results.json")

    return metrics
