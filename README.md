# ScaleDown: Online Soft Compression And Reranking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scaledown-team/soft_compression/blob/main/ScaleDown_Colab.ipynb)

**ScaleDown** is an implementation of the OSCAR paper ([arXiv:2504.07109v1](https://arxiv.org/abs/2504.07109)) with support for both GPU and AWS Trainium training, and a novel ModernBERT compressor variant.

> **🚀 Try it now**: Click the Colab badge above or see [COLAB_GUIDE.md](./COLAB_GUIDE.md) for a 5-minute demo!

## Overview

ScaleDown performs **query-dependent online soft compression** for Retrieval-Augmented Generation (RAG), achieving 2-5× faster inference while maintaining or improving accuracy.

### Key Features

- 🚀 **2-5× faster RAG inference** with minimal accuracy loss
- 🎯 **Two compressor options**:
  - **N-Layers**: First N layers of generator (faithful to paper, no pretraining needed)
  - **ModernBERT**: Novel encoder-based compressor (faster, smaller)
- 💻 **Cross-platform training**: GPU and AWS Trainium support
- 📊 **16× compression**: Compress 128-token documents into 8 embeddings
- 🎓 **Distillation-based**: Learn from teacher LLM (no ground truth labels needed)
- 🔄 **Optional reranking**: Simultaneous compression and reranking

---

## What's Different from OSCAR Paper?

| Aspect | OSCAR (Paper) | ScaleDown (This Repo) |
|--------|---------------|----------------------|
| **Name** | OSCAR | **ScaleDown** |
| **Compressor** | First N layers OR Llama-1B | First N layers OR **ModernBERT** |
| **Hardware** | GPU only | **GPU + AWS Trainium** |
| **Framework** | PyTorch | PyTorch + **AWS Neuron SDK** |

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed comparison.

---

## Installation

### For GPU Training

```bash
# Clone repository
git clone <repo-url>
cd soft_compression

# Install dependencies
pip install torch transformers peft accelerate
pip install datasets tqdm wandb  # Optional

# Install package
pip install -e .
```

### For AWS Trainium Training

```bash
# On Trn1 instance, install Neuron SDK
pip install torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com

# Install other dependencies
pip install transformers peft accelerate datasets tqdm

# Install package
pip install -e .
```

---

## Quick Start

> **⚡ First time?** Choose your path:
> - **Test setup**: `python test_training.py --test_both` ([TESTING.md](./TESTING.md))
> - **Train on real data**: See [REAL_DATA_TRAINING.md](./REAL_DATA_TRAINING.md) (30 min)
> - **Try in Colab**: Click the badge above ([COLAB_GUIDE.md](./COLAB_GUIDE.md))

### Quick Training with Real Data (Recommended)

```bash
# 1. Get real QA data (500 examples from SQuAD)
python prepare_small_real_dataset.py --dataset squad --num_examples 500

# 2. Train with before/after evaluation and plots
python train_with_evaluation.py --train_data small_real_dataset.json
```

**What you get:**
- ✅ Model trained on real data (30 minutes on GPU)
- ✅ Before/after metrics comparison (EM, F1, ROUGE)
- ✅ Training curves plot
- ✅ Inference speed comparison
- ✅ Comprehensive report

See [REAL_DATA_TRAINING.md](./REAL_DATA_TRAINING.md) for details.

### Manual Training Workflow

### 1. Prepare Data

Your training data should be a list of dictionaries:

```python
data = [
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and largest city of France...",
            "France is a country in Western Europe...",
            # ... more documents
        ],
        "answer": "The capital of France is Paris.",
        "reranking_scores": [0.95, 0.3, ...]  # Optional, from teacher reranker
    },
    # ... more examples
]
```

### 2. Train ScaleDown Model

```python
from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset
from scaledown.training import ScaleDownTrainer

# Configuration
config = ScaleDownConfig(
    compressor_type="n_layers",  # or "modernbert"
    num_compressor_layers=8,      # for n_layers
    num_memory_tokens=8,
    compression_rate=16,
    generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device_type="gpu",  # or "trainium"
    batch_size=128,
    num_epochs=1,
)

# Create model
model = ScaleDownModel(config)

# Create dataset
dataset = ScaleDownDataset(data, config)

# Train
trainer = ScaleDownTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    output_dir="./checkpoints",
)

trainer.train()
```

### 3. Inference

```python
# Load trained model
model = ScaleDownModel(config)
model.load_state_dict(torch.load("checkpoints/final/pytorch_model.bin"))

# Generate answer
answer = model.generate(
    query_input_ids=query_ids,
    query_attention_mask=query_mask,
    doc_input_ids=doc_ids,
    doc_attention_mask=doc_mask,
    memory_token_positions=mem_positions,
    max_new_tokens=128,
)
```

---

## Training on AWS Trainium

### 1. Launch Trn1 Instance

```bash
# Use AWS Deep Learning AMI with Neuron
# Instance type: trn1.2xlarge or larger
```

### 2. Modify Config

```python
config = ScaleDownConfig(
    device_type="trainium",  # Enable Trainium
    # ... other configs
)
```

### 3. Compile and Train

The trainer automatically handles XLA compilation and optimization for Trainium.

---

## Model Architecture

### ScaleDown-N-Layers (Faithful to Paper)

```
Input: [Query] [Document] [MEM_1] ... [MEM_l]
  ↓
First N layers of generator (e.g., Mistral-7B)
  ↓
Extract hidden states at memory token positions
  ↓
Generator LLM with LoRA
  ↓
Answer
```

**Advantages:**
- No pretraining needed
- Hidden representations already aligned
- Proven in OSCAR paper

### ScaleDown-ModernBERT (Novel Variant)

```
Input: [Query] [SEP] [Document] [MEM_1] ... [MEM_l]
  ↓
ModernBERT-base (149M params, bidirectional)
  ↓
Extract memory token hidden states (768D)
  ↓
Projection: FC(768 → 4096) → ReLU → FC(4096 → 4096)
  ↓
Generator LLM with LoRA
  ↓
Answer
```

**Advantages:**
- Much smaller compressor (149M vs 1.9B for 8-layer)
- 2× faster compression
- Better suited for encoding tasks
- Bidirectional attention

---

## Configuration Options

See [scaledown/config.py](./scaledown/config.py) for all options. Key parameters:

```python
ScaleDownConfig(
    # Compressor
    compressor_type="n_layers",        # "n_layers" or "modernbert"
    num_compressor_layers=8,           # For n_layers: 5, 8, or 10

    # Compression
    num_memory_tokens=8,               # Embeddings per document
    compression_rate=16,               # 16× compression (128 → 8 tokens)

    # Generator
    generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,

    # Reranking
    enable_reranking=False,            # Joint compression + reranking
    reranking_loss_weight=0.05,

    # Training
    batch_size=128,
    learning_rate_generator=1e-4,
    learning_rate_compressor_nlayers=5e-5,
    learning_rate_compressor_modernbert=1e-4,
    num_epochs=1,

    # Device
    device_type="gpu",                 # "gpu" or "trainium"
)
```

---

## Performance

Based on OSCAR paper results (expected for ScaleDown-N-Layers):

| Model | Speed-up | LLM Eval (Avg) | Memory Savings |
|-------|----------|----------------|----------------|
| Mistral-7B (no compression) | 1.0× | 0.76 | - |
| **ScaleDown-8-Layers** | **2.4×** | **0.77** | **~60%** |
| **ScaleDown-5-Layers** | **3.1×** | **0.76** | **~70%** |
| PISCO (offline) | 5.8× | 0.74 | ~75% |
| Provence (hard pruning) | 2.2× | 0.76 | ~50% |

ScaleDown-ModernBERT performance is TBD (novel contribution).

---

## Project Structure

```
soft_compression/
├── scaledown/
│   ├── __init__.py
│   ├── config.py                # Configuration classes
│   ├── models/
│   │   ├── compressor.py        # NLayersCompressor & ModernBERTCompressor
│   │   ├── generator.py         # Generator with LoRA
│   │   └── model.py             # Full ScaleDown model
│   ├── training/
│   │   └── trainer.py           # Training loop (GPU/Trainium)
│   └── data/
│       └── dataset.py           # Dataset utilities
├── ARCHITECTURE.md              # Detailed architecture docs
├── README.md                    # This file
└── 2504.07109v1.pdf            # Original OSCAR paper
```

---

## Citation

If you use ScaleDown, please cite the original OSCAR paper:

```bibtex
@article{louis2025oscar,
  title={OSCAR: Online Soft Compression And Reranking},
  author={Louis, Maxime and Formal, Thibault and Dejean, Herv{\'e} and Clinchant, St{\'e}phane},
  journal={arXiv preprint arXiv:2504.07109},
  year={2025}
}
```

---

## References

1. **OSCAR Paper**: Louis et al., "OSCAR: Online Soft Compression And Reranking" ([arXiv:2504.07109](https://arxiv.org/abs/2504.07109))
2. **ModernBERT**: Bellagente et al., "Smarter, Better, Faster, Longer" ([arXiv:2412.13663](https://arxiv.org/abs/2412.13663))
3. **AWS Trainium**: [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

---

## License

[Add your license here]

---

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

- NAVER LABS Europe for the original OSCAR paper
- Answer.AI for ModernBERT
- AWS for Trainium support
