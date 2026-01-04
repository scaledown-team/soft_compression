# ScaleDown Quick Start Guide

Get up and running with ScaleDown in 5 minutes!

## Installation

```bash
# Clone repository
cd /path/to/soft_compression

# Install dependencies
pip install -e .
```

## Minimal Working Example

```python
import torch
from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset
from scaledown.training import ScaleDownTrainer

# 1. Configure
config = ScaleDownConfig(
    compressor_type="n_layers",      # Use first 8 layers of Mistral-7B
    num_compressor_layers=8,
    compression_rate=16,              # Compress 128 tokens → 8 embeddings
    device_type="gpu",                # or "trainium"
    batch_size=4,                     # Small for demo (paper uses 128)
)

# 2. Prepare data
data = [
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France...",
            "France is in Western Europe...",
            # ... more documents
        ],
        "answer": "The capital of France is Paris.",
    },
    # ... more examples
]

# 3. Create dataset and model
dataset = ScaleDownDataset(data, config)
model = ScaleDownModel(config)

# 4. Train
trainer = ScaleDownTrainer(model, config, dataset)
trainer.train()
```

## Command-Line Training

```bash
# Prepare your data.json:
# [
#   {
#     "query": "...",
#     "documents": ["...", "..."],
#     "answer": "..."
#   },
#   ...
# ]

# Train with N-Layers compressor (no pretraining needed)
python train.py \
  --compressor_type n_layers \
  --num_layers 8 \
  --train_data data.json \
  --output_dir ./checkpoints

# Train with ModernBERT compressor (novel variant)
python train.py \
  --compressor_type modernbert \
  --train_data data.json \
  --output_dir ./checkpoints
```

## Key Configuration Options

```python
ScaleDownConfig(
    # Compressor: "n_layers" (paper) or "modernbert" (novel)
    compressor_type="n_layers",
    num_compressor_layers=8,           # 5, 8, or 10

    # Compression: 16× = 128 tokens → 8 embeddings
    num_memory_tokens=8,
    compression_rate=16,

    # Generator: Any causal LM
    generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",

    # Device: "gpu" or "trainium"
    device_type="gpu",

    # Training (from OSCAR paper)
    batch_size=128,
    learning_rate_generator=1e-4,
    learning_rate_compressor_nlayers=5e-5,

    # Optional: Enable reranking
    enable_reranking=False,
)
```

## Data Format

Your training data should be JSON with this format:

```json
[
  {
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI that enables computers to learn...",
      "There are three main types of machine learning: supervised...",
      "Machine learning algorithms use statistical techniques..."
    ],
    "answer": "Machine learning is a subset of artificial intelligence...",
    "reranking_scores": [0.95, 0.7, 0.5]  // Optional
  }
]
```

### How to Generate Training Data

The ScaleDown implementation includes complete dataset generation utilities following the OSCAR paper:

```bash
# Quick test with synthetic queries
python -m scaledown.data.prepare_dataset \
  --num_synthetic_queries 100 \
  --output_file test_data.json

# Full pipeline following OSCAR paper
python -m scaledown.data.prepare_dataset \
  --download_ms_marco \
  --corpus_path kilt_knowledgesource.json \
  --output_file train_data.json \
  --enable_reranking
```

**See [DATASET_PREPARATION.md](./DATASET_PREPARATION.md) for the complete guide**, including:
- Downloading Wikipedia-KILT corpus
- Using MS MARCO or custom queries
- SPLADE-v3 retrieval
- DeBERTa-v3 reranking
- Mistral-7B teacher generation

## Training on AWS Trainium

```bash
# 1. Launch Trn1 instance (AWS Deep Learning AMI)
# 2. Install Neuron SDK
pip install torch-neuronx neuronx-cc \
  --extra-index-url https://pip.repos.neuron.amazonaws.com

# 3. Train with device=trainium
python train.py \
  --compressor_type n_layers \
  --train_data data.json \
  --device trainium
```

The trainer automatically handles XLA compilation and optimization!

## Comparison: N-Layers vs ModernBERT

| Feature | N-Layers | ModernBERT |
|---------|----------|------------|
| **Faithfulness** | ✅ Exact OSCAR paper | Novel contribution |
| **Pretraining** | ❌ Not needed | ✅ May be needed |
| **Size** | 1.9B params (8-layer) | 149M params |
| **Speed** | Fast | Faster (2× compression) |
| **Attention** | Causal (decoder) | Bidirectional (encoder) |

**Recommendation**: Start with N-Layers (proven, no pretraining). Try ModernBERT as an experiment.

## Next Steps

1. **Read the docs**:
   - [README.md](./README.md) - Full documentation
   - [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical deep dive
   - [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - What we built

2. **Try examples**:
   ```bash
   python example_usage.py
   ```

3. **Prepare your data** using your retriever + teacher LLM

4. **Train and evaluate** on your RAG benchmark

## Troubleshooting

### Out of Memory

```python
config = ScaleDownConfig(
    batch_size=32,              # Reduce from 128
    gradient_checkpointing=True, # Enable (default)
    use_bf16=True,               # Use BF16 (default)
)
```

### Slow Training

- Use fewer compressor layers (5 instead of 8)
- Try ModernBERT (smaller, faster)
- Use Trainium for better cost/performance

### ModernBERT Not Working

ModernBERT may need pretraining (similar to OSCAR-llama in paper). See paper Appendix I for details.

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/scaledown/issues)
- Docs: [README.md](./README.md)
- Paper: [OSCAR arXiv:2504.07109](https://arxiv.org/abs/2504.07109)

---

Happy compressing! 🚀
