# ScaleDown Implementation Summary

## What We Built

A complete implementation of the OSCAR paper with the following enhancements:

### ✅ Core Features (Faithful to Paper)

1. **ScaleDown-N-Layers Compressor**
   - Uses first N layers (5, 8, or 10) of the generator model
   - No pretraining required (hidden representations already aligned)
   - Implements exactly as described in OSCAR paper Section 3

2. **Generator with LoRA**
   - Mistral-7B-Instruct (or any causal LM)
   - LoRA fine-tuning (r=16, alpha=32, dropout=0.1)
   - Exactly as OSCAR paper Table 7

3. **Training Pipeline**
   - Sentence-level distillation from teacher LLM
   - Batch size: 128
   - Learning rates: 1e-4 (generator), 5e-5 (N-Layers compressor)
   - All hyperparameters from OSCAR paper Table 7

4. **Optional Reranking**
   - Joint compression + reranking with RR token
   - L2 loss with λ=0.05
   - Distillation from DeBERTa-v3 (as in paper)

### 🆕 Novel Contributions

1. **ScaleDown-ModernBERT Compressor** (Our Innovation)
   - Uses ModernBERT-base (149M params) instead of Llama-1B (1.1B params)
   - Encoder-only model (bidirectional attention)
   - Projection layer: 768D → 4096D
   - Rationale: Compression is fundamentally an encoding task
   - Expected benefits: 2× faster, smaller model, better encoding

2. **AWS Trainium Support** (Our Innovation)
   - Full support for AWS Trainium instances (Trn1/Trn1n)
   - Native PyTorch backend via TorchNeuron
   - XLA compilation and optimization
   - Device abstraction layer for seamless GPU/Trainium switching

### 📊 Dataset Generation Utilities

1. **Complete Data Pipeline** (Following OSCAR Paper)
   - SPLADE-v3 retrieval from Wikipedia-KILT
   - DeBERTa-v3 reranking (optional)
   - Mistral-7B teacher answer generation
   - MS MARCO query support
   - Automated dataset preparation script

2. **Implementation Files**
   - `scaledown/data/retrieval.py` - SPLADE-v3 sparse retrieval
   - `scaledown/data/teacher.py` - Teacher LLM generation
   - `scaledown/data/reranker.py` - DeBERTa-v3 reranking
   - `scaledown/data/prepare_dataset.py` - End-to-end pipeline

### 📚 Comprehensive Documentation

1. **ARCHITECTURE.md**
   - Detailed comparison: OSCAR paper vs ScaleDown
   - Architecture diagrams
   - Design rationale for ModernBERT
   - Training details and loss functions

2. **README.md**
   - Quick start guide
   - Installation instructions (GPU & Trainium)
   - Code examples
   - Performance benchmarks

3. **DATASET_PREPARATION.md**
   - Complete dataset generation guide
   - Wikipedia-KILT setup
   - MS MARCO queries
   - Custom corpus usage

## File Structure

```
soft_compression/
├── scaledown/                      # Main package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # ScaleDownConfig with all hyperparameters
│   ├── models/
│   │   ├── __init__.py
│   │   ├── compressor.py          # NLayersCompressor & ModernBERTCompressor
│   │   ├── generator.py           # Generator with LoRA & device abstraction
│   │   └── model.py               # ScaleDownModel (full pipeline)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop (GPU/Trainium support)
│   └── data/
│       ├── __init__.py
│       ├── dataset.py             # ScaleDownDataset with memory tokens
│       ├── retrieval.py           # SPLADE-v3 document retrieval
│       ├── teacher.py             # Teacher LLM answer generation
│       ├── reranker.py            # DeBERTa-v3 reranking
│       └── prepare_dataset.py     # End-to-end dataset generation
│
├── ARCHITECTURE.md                # Detailed architecture documentation
├── README.md                      # User guide and API docs
├── IMPLEMENTATION_SUMMARY.md      # This file
├── DATASET_PREPARATION.md         # Dataset generation guide
├── QUICKSTART.md                  # Quick start guide
├── train.py                       # Main training script
├── example_usage.py               # Example code
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
└── 2504.07109v1.pdf              # Original OSCAR paper
```

## Key Implementation Details

### 1. Compressor Architecture

#### ScaleDown-N-Layers (OSCAR faithful)
```python
class NLayersCompressor:
    - Extracts first N layers from generator
    - Processes: [Query] [Document] [MEM_1] ... [MEM_l]
    - Returns: [batch, num_mem_tokens, hidden_size]
    - No projection needed (already aligned)
```

#### ScaleDown-ModernBERT (Novel)
```python
class ModernBERTCompressor:
    - ModernBERT-base encoder (22 layers, 768D)
    - Processes: [Query] [SEP] [Document] [MEM_1] ... [MEM_l]
    - Projection: FC(768→4096) → ReLU → FC(4096→4096)
    - Returns: [batch, num_mem_tokens, 4096]
```

### 2. Generator Architecture

```python
class ScaleDownGenerator:
    - Base: Mistral-7B-Instruct (or configurable)
    - LoRA adapters on all linear layers
    - Input: [Query tokens] + [Compressed embeddings]
    - Device abstraction: GPU (CUDA) or Trainium (XLA)
```

### 3. Training Loss

```python
L(C, G) = -Σ log G(a_i | query, c_1, ..., c_k, a_{<i})  # Generation
          + λ Σ (r_i - r'_i)²                            # Reranking (optional)
```

Where:
- First term: Cross-entropy for generation (distillation from teacher)
- Second term: L2 for reranking scores (distillation from DeBERTa-v3)
- λ = 0.05

### 4. Memory Token Mechanism

Documents are preprocessed as:
```
[Document tokens (120)] [MEM_1] [MEM_2] ... [MEM_8] [RR]?
                         ↑                           ↑
                         Extracted for compression  Optional reranking
```

Compressor extracts hidden states at MEM token positions → compressed embeddings

### 5. Device Abstraction

```python
if device_type == "gpu":
    device = torch.device("cuda")
    # Standard PyTorch training
elif device_type == "trainium":
    import torch_neuronx
    device = torch.device("xla")
    # XLA compilation for Trainium
```

## Usage Examples

### Training with N-Layers (Faithful to Paper)

```bash
python train.py \
  --compressor_type n_layers \
  --num_layers 8 \
  --train_data data.json \
  --device gpu
```

### Training with ModernBERT (Novel)

```bash
python train.py \
  --compressor_type modernbert \
  --train_data data.json \
  --device gpu
```

### Training on AWS Trainium

```bash
python train.py \
  --compressor_type n_layers \
  --num_layers 8 \
  --train_data data.json \
  --device trainium
```

### With Reranking

```bash
python train.py \
  --compressor_type n_layers \
  --enable_reranking \
  --train_data data.json
```

## Testing the Implementation

```python
from scaledown import ScaleDownConfig, ScaleDownModel

# Create config
config = ScaleDownConfig(
    compressor_type="n_layers",
    num_compressor_layers=8,
)

# Initialize model
model = ScaleDownModel(config)

# Model is ready for training or inference
```

## What Makes This Different from OSCAR

| Feature | OSCAR Paper | ScaleDown (This Implementation) |
|---------|-------------|--------------------------------|
| **Name** | OSCAR | **ScaleDown** |
| **Compressor 1** | First N layers of generator | ✅ Same (ScaleDown-N-Layers) |
| **Compressor 2** | Llama-3.2-1B (1.1B params) | ✅ **ModernBERT-base (149M params)** |
| **Compressor Type** | Decoder-only | N-Layers: Decoder<br>**ModernBERT: Encoder** |
| **Hardware** | GPU only | ✅ **GPU + AWS Trainium** |
| **Framework** | PyTorch | ✅ **PyTorch + Neuron SDK** |
| **Attention** | Causal (decoder) | N-Layers: Causal<br>**ModernBERT: Bidirectional** |

## Expected Performance

### ScaleDown-N-Layers (based on OSCAR results)

| Variant | Speed-up | Accuracy | Memory Savings |
|---------|----------|----------|----------------|
| 5-Layers | 3.1× | ~0.76 | ~70% |
| 8-Layers | 2.4× | ~0.77 | ~60% |
| 10-Layers | 2.2× | ~0.77 | ~50% |

### ScaleDown-ModernBERT (predicted)

- **Speed-up**: 3.0-3.5× (smaller, faster compressor)
- **Accuracy**: TBD (requires training to validate)
- **Memory**: ~70-80% savings
- **Size**: 149M params vs 1.9B (8-layer) or 1.1B (Llama-1B)

## Next Steps for Users

1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Prepare your data** (see README.md for format)

3. **Run training**
   ```bash
   python train.py --train_data your_data.json
   ```

4. **Evaluate** on your RAG benchmark

5. **Compare** N-Layers vs ModernBERT performance

## Research Questions to Explore

1. Does ModernBERT's bidirectional attention help compression quality?
2. How does ModernBERT compare to Llama-1B on speed and accuracy?
3. Does pretraining help ModernBERT (like it does for Llama-1B)?
4. How well does this scale to longer contexts (>128 tokens)?
5. Can we achieve better compression rates (32×, 64×)?

## Credits

- **Original OSCAR paper**: Louis et al., NAVER LABS Europe
- **ModernBERT**: Answer.AI team
- **AWS Trainium support**: AWS Neuron SDK team
- **Implementation**: This repository

## License

[Specify your license]
