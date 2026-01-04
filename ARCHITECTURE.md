# ScaleDown Architecture Documentation

## Overview

ScaleDown is based on the OSCAR (Online Soft Compression And Reranking) paper from NAVER LABS Europe (arXiv:2504.07109v1). This document details what the original paper proposed and how our implementation differs.

---

## Original OSCAR Paper Architecture

### Core Concept
OSCAR performs **query-dependent online soft compression** for Retrieval-Augmented Generation (RAG). Unlike offline compression methods, OSCAR compresses documents at inference time based on the current query.

### Compressor Architectures (Paper)

The paper proposed **two compressor variants**:

#### 1. OSCAR-N-Layers
- **Architecture**: Headless transformer using first N layers of the generator backbone
- **Tested configurations**: N = 5, 8, 10 layers (from Mistral-7B, Qwen-7B, etc.)
- **Key advantage**: No pretraining needed - hidden representations already aligned
- **Model size**: ~1.2B params (5 layers), ~1.9B params (8 layers) from 7B model
- **Training**: Direct fine-tuning on distillation data

#### 2. OSCAR-llama
- **Architecture**: Llama-3.2-1B (full decoder-only LLM) as compressor
- **Mapping layer**: 2 fully connected layers with ReLU to map compressor hidden space (2048D) to generator hidden space (4096D for Mistral-7B)
- **Model size**: 1.1B parameters
- **Training**: Requires pretraining on auto-encoding and text continuation tasks using FineWeb data
- **Key advantage**: Faster compression than N-Layers variants

### Generator Architecture (Paper)

- **Models tested**: Mistral-7B, Qwen-7B, Llama-1B, Mistral-24B
- **Fine-tuning method**: LoRA adapters
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: all-linear layers
- **Training objective**: Sentence-level distillation from teacher LLM (Mistral-7B-Instruct)

### Compression Process (Paper)

1. **Input**: Query q, Document d_i, Memory tokens MEM_{1...l}
2. **Compression**: `(c_1^i, ..., c_l^i) = Compressor([Query] [Document] MEM_1 MEM_2 ... MEM_l)`
3. **Output**: l embedding vectors per document (l=8 for 16x compression of 128-token docs)
4. **Generation**: Generator receives query + compressed embeddings from all documents

### Reranking Component (Paper)

- **Token**: Special `RR` token added to compressor input
- **Architecture**: Dense layer maps RR token hidden state to relevance score
- **Training**: Pointwise distillation from DeBERTa-v3 cross-encoder
- **Loss weight (λ)**: 0.05

### Training Details (Paper)

- **Dataset**: 893k queries (393k from prior work + 500k MS MARCO)
- **Document collection**: Wikipedia-KILT, chunks of 128 tokens
- **Retrieval**: SPLADE-v3
- **Reranking**: DeBERTa-v3
- **Documents per query (training)**: 5
- **Documents per query (inference)**: 10
- **Teacher model**: Mistral-7B-Instruct-v0.2
- **Compression rate**: 16x (8 memory embeddings per 128-token doc)
- **Batch size**: 128
- **Epochs**: 1
- **Learning rates**:
  - Generator (LoRA): 1e-4
  - Llama compressor: 1e-4
  - N-Layers compressor: 5e-5
- **Optimizer**: AdamW with linear LR scheduler

---

## ScaleDown Implementation

### Key Differences from OSCAR

| Aspect | OSCAR (Paper) | ScaleDown (Our Implementation) |
|--------|---------------|--------------------------------|
| **Compressor Options** | 1. First N layers of generator<br>2. Llama-3.2-1B | 1. First N layers of generator<br>2. ModernBERT-base (novel) |
| **Compressor Type** | Decoder-only transformers | N-Layers: Decoder-only<br>ModernBERT: Encoder-only |
| **Model Name** | OSCAR | **ScaleDown** |
| **Hardware Support** | GPU only | **GPU + AWS Trainium** |
| **Training Framework** | Standard PyTorch | PyTorch + AWS Neuron SDK (TorchNeuron) |

### Why ModernBERT?

We introduce **ScaleDown-ModernBERT** as a novel compressor variant:

**Rationale**:
1. **Task alignment**: Compressing query-document pairs is fundamentally an encoding task
2. **Efficiency**: ModernBERT-base (149M params) is much smaller than Llama-1B (1.1B params)
3. **Speed**: ModernBERT processes tokens ~2x faster than decoder-only models
4. **Context length**: Supports 8,192 tokens (vs 128k for Llama but we only need 256)
5. **Modern architecture**: Uses RoPE, Flash Attention, GeGLU - similar to modern LLMs
6. **Precedent**: OSCAR already uses DeBERTa-v3 (encoder) for reranking successfully

**Architecture details**:
- **Model**: ModernBERT-base (answerdotai/ModernBERT-base)
- **Parameters**: 149M
- **Hidden size**: 768D
- **Layers**: 22
- **Attention heads**: 12
- **Context**: 8,192 tokens
- **Mapping layer**: 2 FC layers with ReLU (768D → 4096D for Mistral-7B)

### AWS Trainium Support

ScaleDown adds support for AWS Trainium instances (Trn1/Trn1n):

**Implementation approach**:
1. **Device abstraction**: Automatic device selection (GPU/Trainium)
2. **Neuron SDK integration**: Uses AWS Neuron 2.27+ with native PyTorch support
3. **TorchNeuron backend**: Minimal code changes via `torch_neuronx`
4. **Distributed training**: Supports neuron_parallel_compile for multi-chip training
5. **XLA compilation**: Automatic graph optimization for Trainium

**Code structure**:
```python
if config.device_type == "trainium":
    import torch_neuronx
    device = torch.device("xla")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Compressor Variants

#### ScaleDown-N-Layers (Faithful to paper)
```
Input: [Query tokens] [Document tokens] [MEM_1] [MEM_2] ... [MEM_l]
       ↓
First N layers of generator (e.g., Mistral-7B)
       ↓
Extract hidden states of memory tokens
       ↓
Output: l compressed embeddings (768D each)
```

#### ScaleDown-ModernBERT (Novel)
```
Input: [Query tokens] [SEP] [Document tokens] [MEM_1] [MEM_2] ... [MEM_l]
       ↓
ModernBERT-base (22 layers, bidirectional attention)
       ↓
Extract hidden states of memory tokens (768D)
       ↓
Projection: FC(768 → 4096) → ReLU → FC(4096 → 4096)
       ↓
Output: l compressed embeddings (4096D each)
```

### Training Differences

| Component | OSCAR | ScaleDown |
|-----------|-------|-----------|
| **Pretraining** | Only for OSCAR-llama | ScaleDown-N-Layers: No<br>ScaleDown-ModernBERT: Yes (similar to OSCAR-llama) |
| **Device support** | GPU | GPU + Trainium (via Neuron SDK) |
| **Gradient checkpointing** | Not mentioned | Enabled for memory efficiency |
| **Mixed precision** | Not specified | BF16 on GPU, automatic on Trainium |
| **Distributed training** | Standard DDP | DDP (GPU) + neuron_parallel_compile (Trainium) |

### Generator Architecture

Same as OSCAR paper:
- **Base model**: Mistral-7B-Instruct-v0.2 (configurable)
- **Fine-tuning**: LoRA adapters
- **Parameters**: Same as paper (r=16, alpha=32, dropout=0.1)

### Loss Function

Same as OSCAR:
```
L(C, G) = -Σ log G(a_i | query, c_1, ..., c_k, a_{<i})
          + λ Σ (r_i - r'_i)²
```

Where:
- First term: Cross-entropy loss for generation (distillation from teacher)
- Second term: L2 loss for reranking scores (distillation from DeBERTa-v3)
- λ = 0.05 (reranking loss weight)

---

## Performance Expectations

Based on OSCAR paper results, we expect:

### ScaleDown-N-Layers
- **Speed-up**: 2.2-2.4× (8 layers) to 3.1× (5 layers)
- **Accuracy**: On par with or slightly better than uncompressed baseline
- **Memory savings**: ~50-75%
- **No pretraining required**

### ScaleDown-ModernBERT (Predicted)
- **Speed-up**: 3.0-3.5× (smaller compressor)
- **Accuracy**: TBD (may match or exceed OSCAR-llama)
- **Memory savings**: ~60-80%
- **Requires pretraining**

### Trainium-Specific Benefits
- **Cost**: ~50% lower than equivalent GPU instances
- **Scalability**: Better for large-scale pretraining
- **Performance**: Comparable to GPUs with XLA optimization

---

## References

1. **OSCAR Paper**: Louis et al., "OSCAR: Online Soft Compression And Reranking" (arXiv:2504.07109v1)
2. **ModernBERT**: Bellagente et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder" (arXiv:2412.13663)
3. **AWS Trainium**: AWS Neuron SDK 2.27+ Documentation
4. **DeBERTa-v3**: He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training"

---

## Implementation Status

- [x] Architecture design
- [x] Documentation
- [ ] ScaleDown-N-Layers compressor
- [ ] ScaleDown-ModernBERT compressor
- [ ] Generator with LoRA
- [ ] Training loop (GPU)
- [ ] Training loop (Trainium)
- [ ] Reranking component
- [ ] Data loading pipeline
- [ ] Evaluation scripts
