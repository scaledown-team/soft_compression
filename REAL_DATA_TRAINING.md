## Training ScaleDown with Real Data - Quick Start

This guide shows how to train ScaleDown on real QA data with automatic before/after evaluation and performance plots.

### 🎯 TL;DR - One Command

```bash
# First time: Install dependencies (no package installation!)
pip install -r requirements.txt

# Download real data (500 examples from SQuAD)
python prepare_small_real_dataset.py --dataset squad --num_examples 500

# Train with evaluation and plotting
python train_with_evaluation.py --train_data small_real_dataset.json
```

Done! You'll get:
- ✅ Model trained on real data
- ✅ Before/after metrics comparison
- ✅ Training curves plot
- ✅ Inference speed comparison
- ✅ Comprehensive JSON report

---

## Step-by-Step Instructions

### Step 1: Prepare Real Dataset (2 minutes)

Choose one of these real-world QA datasets:

**Option A: SQuAD (Recommended - Easy)**
```bash
python prepare_small_real_dataset.py \
  --dataset squad \
  --num_examples 500 \
  --output_file small_real_dataset.json
```

**Option B: TriviaQA (Good for multi-document RAG)**
```bash
python prepare_small_real_dataset.py \
  --dataset trivia_qa \
  --num_examples 500 \
  --output_file small_real_dataset.json
```

**Option C: HotpotQA (Multi-hop reasoning)**
```bash
python prepare_small_real_dataset.py \
  --dataset hotpot_qa \
  --num_examples 500 \
  --output_file small_real_dataset.json
```

**Option D: Mixed Dataset (All of the above)**
```bash
python prepare_small_real_dataset.py \
  --dataset all \
  --num_examples 800 \
  --output_file small_real_dataset.json
```

This creates:
- `small_real_dataset.json` - 450 training examples (90%)
- `small_real_dataset_eval.json` - 50 eval examples (10%)

### Step 2: Train with Evaluation (30 minutes on GPU)

**Basic Training:**
```bash
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type modernbert \
  --batch_size 4 \
  --num_epochs 1
```

**For Colab Free Tier (T4 GPU):**
```bash
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type modernbert \
  --batch_size 2 \
  --num_epochs 1 \
  --output_dir ./training_results
```

**For Better GPU (A100):**
```bash
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type n_layers \
  --num_layers 8 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir ./training_results
```

### Step 3: View Results

After training completes, check the `training_with_eval/` directory:

```
training_with_eval/
├── before_after_comparison.png     # Bar chart: Before vs After metrics
├── training_curves.png             # Training loss over time
├── speed_comparison.png            # Inference speed comparison
├── metrics_report.json             # Detailed metrics (EM, F1, ROUGE)
├── before_training_results.json    # Baseline performance
├── after_training_results.json     # Final performance
└── checkpoints/                    # Model checkpoints
    └── final/                      # Final trained model
```

**View plots:**
```bash
# On local machine
open training_with_eval/before_after_comparison.png
open training_with_eval/training_curves.png
open training_with_eval/speed_comparison.png

# In Colab
from IPython.display import Image, display
display(Image('training_with_eval/before_after_comparison.png'))
display(Image('training_with_eval/training_curves.png'))
display(Image('training_with_eval/speed_comparison.png'))
```

**Read metrics:**
```bash
cat training_with_eval/metrics_report.json
```

---

## What Gets Evaluated

### Quality Metrics

1. **Exact Match (EM)**
   - Perfect match between prediction and ground truth
   - Range: 0-1 (higher is better)

2. **Token F1**
   - Word-level overlap between prediction and ground truth
   - Range: 0-1 (higher is better)

3. **ROUGE-L**
   - Longest common subsequence F1
   - Range: 0-1 (higher is better)

### Speed Metrics

1. **Mean Inference Time** - Average time per query (ms)
2. **Throughput** - Queries per second
3. **Speedup** - After/Before ratio

---

## Expected Results

### SQuAD Dataset (500 examples)

**Before Training (Random Initialization):**
- Exact Match: ~0.00
- Token F1: ~0.10-0.15
- ROUGE-L: ~0.10-0.15
- Inference: ~50-100ms/query

**After Training (1 epoch):**
- Exact Match: ~0.15-0.25
- Token F1: ~0.35-0.45
- ROUGE-L: ~0.40-0.50
- Inference: ~45-90ms/query (similar or slightly faster)

**Note:** These are conservative estimates for 500 examples. With more data (5k+), you'll see much better results!

### TriviaQA/HotpotQA

Typically harder than SQuAD due to multi-hop reasoning:
- Exact Match: ~0.10-0.20 (after 1 epoch)
- Token F1: ~0.30-0.40
- ROUGE-L: ~0.35-0.45

---

## Visualization Examples

### Before/After Comparison Plot

Shows 3 metrics side-by-side:
- Green bars = After training
- Red bars = Before training
- Percentage improvement shown above

### Training Curves Plot

Shows:
- Training loss over steps (should decrease)
- Smoothed loss curve
- Loss reduction indicates learning

### Speed Comparison Plot

Shows:
- Inference time distribution
- Average speed comparison
- Speedup calculation

---

## Configuration Options

### Model Options

```bash
# Use ModernBERT (faster, smaller)
--compressor_type modernbert

# Use N-Layers (faithful to OSCAR paper)
--compressor_type n_layers --num_layers 8

# Adjust compression
--num_memory_tokens 8 --compression_rate 16  # 16× compression
--num_memory_tokens 4 --compression_rate 8   # 8× compression
```

### Training Options

```bash
# Quick test (100 steps)
--max_steps 100

# Full epoch
--num_epochs 1

# Multiple epochs
--num_epochs 3

# Larger batch (if GPU allows)
--batch_size 8

# Learning rate
--learning_rate 1e-4  # Default
--learning_rate 5e-5  # More conservative
```

### Evaluation Options

```bash
# Evaluation frequency during training
--eval_steps 100  # Eval every 100 steps

# Logging frequency
--logging_steps 10  # Log every 10 steps
```

---

## Scaling Up

### 1. More Training Data

```bash
# Get more examples (1000-5000)
python prepare_small_real_dataset.py \
  --dataset squad \
  --num_examples 5000 \
  --output_file medium_dataset.json

python train_with_evaluation.py \
  --train_data medium_dataset.json \
  --num_epochs 3
```

### 2. Full OSCAR Pipeline

Use Wikipedia-KILT retrieval + teacher LLM (see DATASET_PREPARATION.md):

```bash
# Generate dataset with retrieval
python -m scaledown.data.prepare_dataset \
  --download_ms_marco \
  --corpus_path kilt_knowledgesource.json \
  --output_file full_dataset.json \
  --enable_reranking

# Train
python train_with_evaluation.py \
  --train_data full_dataset.json \
  --num_epochs 1
```

### 3. Longer Training

```bash
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --num_epochs 5 \
  --eval_steps 50 \
  --output_dir ./extended_training
```

---

## Comparing Compressor Types

Test both N-Layers and ModernBERT:

```bash
# Test ModernBERT
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type modernbert \
  --output_dir ./results_modernbert

# Test N-Layers
python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type n_layers \
  --num_layers 8 \
  --output_dir ./results_nlayers

# Compare results
echo "ModernBERT:"
cat results_modernbert/metrics_report.json | grep -A 3 "after_training"

echo "\nN-Layers:"
cat results_nlayers/metrics_report.json | grep -A 3 "after_training"
```

---

## Troubleshooting

### Out of Memory

```bash
# Use smaller batch
--batch_size 1

# Use ModernBERT (uses less memory)
--compressor_type modernbert

# Reduce memory tokens
--num_memory_tokens 4
```

### Training Too Slow

```bash
# Use fewer training examples
python prepare_small_real_dataset.py --num_examples 200

# Limit training steps
--max_steps 500

# Use ModernBERT (2× faster)
--compressor_type modernbert
```

### Low Metrics After Training

This is normal with small datasets (500 examples). To improve:

1. **More data**: Use 5000+ examples
2. **More epochs**: Train for 3-5 epochs
3. **Better data**: Use retrieval + teacher LLM (OSCAR pipeline)
4. **Tune hyperparameters**: Try different learning rates

---

## Colab Integration

Add this to the Colab notebook:

```python
# 1. Prepare data
!python prepare_small_real_dataset.py \
  --dataset squad \
  --num_examples 500

# 2. Train with evaluation
!python train_with_evaluation.py \
  --train_data small_real_dataset.json \
  --compressor_type modernbert \
  --batch_size 2 \
  --num_epochs 1 \
  --output_dir /content/drive/MyDrive/scaledown_training

# 3. View results
from IPython.display import Image, display

print("Before vs After Comparison:")
display(Image('/content/drive/MyDrive/scaledown_training/before_after_comparison.png'))

print("\nTraining Curves:")
display(Image('/content/drive/MyDrive/scaledown_training/training_curves.png'))

print("\nSpeed Comparison:")
display(Image('/content/drive/MyDrive/scaledown_training/speed_comparison.png'))

# 4. Read metrics
import json
with open('/content/drive/MyDrive/scaledown_training/metrics_report.json') as f:
    report = json.load(f)

print("\nMetrics Report:")
print(json.dumps(report, indent=2))
```

---

## Next Steps

1. **Experiment with hyperparameters**:
   - Different compression rates
   - Different learning rates
   - Different batch sizes

2. **Try different datasets**:
   - SQuAD vs TriviaQA vs HotpotQA
   - Mixed datasets

3. **Scale up**:
   - More training examples (5k-50k)
   - Full OSCAR pipeline with retrieval
   - Multiple epochs

4. **Production deployment**:
   - Train final model
   - Benchmark on your RAG task
   - Deploy with compression

---

**Ready to start?**

```bash
# One command to get started
python prepare_small_real_dataset.py --dataset squad --num_examples 500 && \
python train_with_evaluation.py --train_data small_real_dataset.json
```

Then check `training_with_eval/` for all results and plots! 🚀
