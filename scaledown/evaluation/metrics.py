"""
Evaluation metrics and visualization for ScaleDown.

This module provides:
- RAG-specific metrics (exact match, F1, ROUGE, etc.)
- Performance comparison (before vs after training)
- Training curve visualization
"""

import json
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = pred_tokens & gt_tokens

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score (0 or 1).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """
    Compute ROUGE-L score (simplified implementation).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # Compute LCS length
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gt_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_rag_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Compute RAG evaluation metrics.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Dictionary of metrics
    """
    assert len(predictions) == len(ground_truths), "Lengths must match"

    metrics = {
        'exact_match': [],
        'token_f1': [],
        'rouge_l': [],
    }

    for pred, gt in zip(predictions, ground_truths):
        metrics['exact_match'].append(compute_exact_match(pred, gt))
        metrics['token_f1'].append(compute_token_f1(pred, gt))
        metrics['rouge_l'].append(compute_rouge_l(pred, gt))

    # Average metrics
    avg_metrics = {
        f'{key}_avg': np.mean(values) for key, values in metrics.items()
    }

    return avg_metrics


def plot_training_curves(
    train_losses: List[float],
    eval_metrics: Optional[List[Dict[str, float]]] = None,
    output_path: str = "./training_curves.png",
    title: str = "ScaleDown Training",
):
    """
    Plot training curves (loss and metrics).

    Args:
        train_losses: List of training losses
        eval_metrics: List of evaluation metrics (one dict per eval step)
        output_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if eval_metrics and len(eval_metrics) > 0:
        # Extract metric histories
        steps = list(range(0, len(train_losses), len(train_losses) // len(eval_metrics)))[:len(eval_metrics)]

        exact_match = [m.get('exact_match_avg', 0) for m in eval_metrics]
        token_f1 = [m.get('token_f1_avg', 0) for m in eval_metrics]
        rouge_l = [m.get('rouge_l_avg', 0) for m in eval_metrics]

        # Plot 2: Exact Match
        ax = axes[0, 1]
        ax.plot(steps, exact_match, label='Exact Match', color='green', linewidth=2, marker='o')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.set_title('Exact Match')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Token F1
        ax = axes[1, 0]
        ax.plot(steps, token_f1, label='Token F1', color='orange', linewidth=2, marker='o')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.set_title('Token F1')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 4: ROUGE-L
        ax = axes[1, 1]
        ax.plot(steps, rouge_l, label='ROUGE-L', color='red', linewidth=2, marker='o')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        ax.set_title('ROUGE-L')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        # No eval metrics, just show loss in different views
        ax = axes[0, 1]
        ax.plot(train_losses, label='Training Loss (smoothed)', color='blue', linewidth=2, alpha=0.5)
        # Add moving average
        if len(train_losses) > 10:
            window = min(50, len(train_losses) // 10)
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(train_losses)), smoothed, label=f'Moving Avg (window={window})',
                   color='darkblue', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Smoothed)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Clear unused plots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 0].text(0.5, 0.5, 'No evaluation data', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_path}")
    plt.close()


def plot_before_after_comparison(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    output_path: str = "./before_after_comparison.png",
    title: str = "Performance: Before vs After Training",
):
    """
    Plot before/after training comparison.

    Args:
        before_metrics: Metrics before training
        after_metrics: Metrics after training
        output_path: Path to save figure
        title: Plot title
    """
    metrics = ['exact_match_avg', 'token_f1_avg', 'rouge_l_avg']
    metric_names = ['Exact Match', 'Token F1', 'ROUGE-L']

    before_values = [before_metrics.get(m, 0) for m in metrics]
    after_values = [after_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, before_values, width, label='Before Training',
                   color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='After Training',
                   color='lightgreen', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

    # Add improvement percentages
    for i, (before, after) in enumerate(zip(before_values, after_values)):
        if before > 0:
            improvement = ((after - before) / before) * 100
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, max(before, after) + 0.05,
                   f'{improvement:+.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_path}")
    plt.close()


def plot_inference_speed_comparison(
    baseline_times: List[float],
    compressed_times: List[float],
    output_path: str = "./inference_speed_comparison.png",
    title: str = "Inference Speed: Baseline vs ScaleDown",
):
    """
    Plot inference speed comparison.

    Args:
        baseline_times: List of inference times for baseline (ms)
        compressed_times: List of inference times for ScaleDown (ms)
        output_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: Distribution comparison
    ax = axes[0]
    ax.hist(baseline_times, bins=20, alpha=0.6, label='Baseline', color='red')
    ax.hist(compressed_times, bins=20, alpha=0.6, label='ScaleDown', color='green')
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Average comparison
    ax = axes[1]
    avg_baseline = np.mean(baseline_times)
    avg_compressed = np.mean(compressed_times)
    speedup = avg_baseline / avg_compressed

    bars = ax.bar(['Baseline', 'ScaleDown'], [avg_baseline, avg_compressed],
                  color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Average Inference Time (ms)')
    ax.set_title(f'Average Speed (Speedup: {speedup:.2f}×)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f} ms',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Speed comparison plot saved to {output_path}")
    plt.close()


def save_metrics_report(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    speed_stats: Optional[Dict[str, float]] = None,
    output_path: str = "./metrics_report.json",
):
    """
    Save comprehensive metrics report as JSON.

    Args:
        before_metrics: Metrics before training
        after_metrics: Metrics after training
        speed_stats: Speed statistics (optional)
        output_path: Path to save report
    """
    report = {
        "before_training": before_metrics,
        "after_training": after_metrics,
        "improvements": {
            key: {
                "absolute": after_metrics.get(key, 0) - before_metrics.get(key, 0),
                "relative": ((after_metrics.get(key, 0) - before_metrics.get(key, 0)) /
                            before_metrics.get(key, 0.001)) * 100  # Avoid division by zero
            }
            for key in before_metrics.keys()
        }
    }

    if speed_stats:
        report["speed_statistics"] = speed_stats

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Metrics report saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print("\nBefore Training:")
    for key, value in before_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAfter Training:")
    for key, value in after_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nImprovements:")
    for key in before_metrics.keys():
        absolute = report["improvements"][key]["absolute"]
        relative = report["improvements"][key]["relative"]
        print(f"  {key}: {absolute:+.4f} ({relative:+.2f}%)")

    if speed_stats:
        print("\nSpeed Statistics:")
        for key, value in speed_stats.items():
            print(f"  {key}: {value:.2f}")

    print("=" * 80)
