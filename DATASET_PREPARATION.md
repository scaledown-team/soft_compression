# ScaleDown Dataset Preparation Guide

This guide explains how to create training data for ScaleDown following the OSCAR paper's methodology.

## Overview

The OSCAR paper uses a **distillation-based** approach that doesn't require ground-truth labels:

1. **Retrieve** documents for queries using SPLADE-v3
2. **Rerank** documents using DeBERTa-v3 cross-encoder (optional)
3. **Generate** answers using teacher LLM (Mistral-7B)
4. **Train** student model to mimic teacher with compressed documents

The student learns to compress documents while maintaining the ability to generate the same answers as the teacher.

## Quick Start

### Option 1: Synthetic Data (for testing)

```bash
# Generate 100 synthetic examples for quick testing
python -m scaledown.data.prepare_dataset \
  --num_synthetic_queries 100 \
  --output_file test_data.json
```

⚠️ **Note**: This will fail without a Wikipedia-KILT corpus. For pure testing, see "Mock Dataset" below.

### Option 2: MS MARCO Queries (reproducing the paper)

```bash
# Download MS MARCO dev queries and prepare dataset
python -m scaledown.data.prepare_dataset \
  --download_ms_marco \
  --corpus_path /path/to/kilt_knowledgesource.json \
  --output_file train_data.json \
  --top_k_retrieval 20 \
  --top_k_reranking 5 \
  --enable_reranking
```

### Option 3: Custom Queries

```bash
# Prepare queries.txt with one query per line:
# What is machine learning?
# How does photosynthesis work?
# ...

python -m scaledown.data.prepare_dataset \
  --queries_file queries.txt \
  --corpus_path /path/to/kilt_knowledgesource.json \
  --output_file train_data.json
```

## Detailed Pipeline

### Step 1: Download Wikipedia-KILT Corpus

The OSCAR paper uses Wikipedia-KILT as the document corpus.

```bash
# Download Wikipedia-KILT (35GB compressed)
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json

# The corpus contains ~5.9M Wikipedia articles
# Format: {"wikipedia_id": ..., "wikipedia_title": ..., "text": [...]}
```

**Alternative**: For smaller-scale experiments, use a subset:

```python
# Create a smaller corpus for testing
import json

input_file = "kilt_knowledgesource.json"
output_file = "kilt_small.json"
max_docs = 100000

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for i, line in enumerate(fin):
        if i >= max_docs:
            break
        fout.write(line)
```

### Step 2: Prepare Queries

#### Option A: MS MARCO (paper's approach)

The paper uses 893k queries:
- MS MARCO dev queries (6,980)
- PISCO queries (886k+)

Download MS MARCO:

```bash
# Automatically downloaded by prepare_dataset.py with --download_ms_marco
# Or manually:
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.dev.small.tsv
```

For PISCO queries, see: https://github.com/kyunghyuncho/pisco

#### Option B: Existing QA Datasets

Use queries from existing datasets:

```python
from datasets import load_dataset

# Natural Questions
nq = load_dataset("natural_questions", split="train")
queries = [ex["question"]["text"] for ex in nq]

# TriviaQA
trivia = load_dataset("trivia_qa", "unfiltered", split="train")
queries = [ex["question"] for ex in trivia]

# SQuAD
squad = load_dataset("squad", split="train")
queries = [ex["question"] for ex in squad]

# Save
with open("queries.txt", "w") as f:
    for q in queries:
        f.write(q + "\n")
```

#### Option C: Synthetic Queries

Generate domain-specific queries:

```python
# Example: Generate science questions
queries = [
    "What is photosynthesis?",
    "How does DNA replication work?",
    "What causes earthquakes?",
    # ... more questions
]

with open("queries.txt", "w") as f:
    for q in queries:
        f.write(q + "\n")
```

### Step 3: Run Dataset Preparation

#### Basic Pipeline (no reranking)

```bash
python -m scaledown.data.prepare_dataset \
  --queries_file queries.txt \
  --corpus_path kilt_knowledgesource.json \
  --output_file train_data.json \
  --top_k_retrieval 5
```

#### Full Pipeline (with reranking)

```bash
python -m scaledown.data.prepare_dataset \
  --queries_file queries.txt \
  --corpus_path kilt_knowledgesource.json \
  --output_file train_data.json \
  --top_k_retrieval 20 \
  --top_k_reranking 5 \
  --enable_reranking
```

This will:
1. Retrieve 20 documents per query with SPLADE-v3
2. Rerank and filter to top-5 documents
3. Generate answers with Mistral-7B
4. Save reranking scores for joint training

#### Memory-Efficient Options

For limited GPU memory:

```bash
python -m scaledown.data.prepare_dataset \
  --queries_file queries.txt \
  --corpus_path kilt_knowledgesource.json \
  --output_file train_data.json \
  --teacher_8bit \  # Use 8-bit quantization for teacher LLM
  --max_corpus_size 100000  # Use smaller corpus subset
```

### Step 4: Verify Output Format

The output JSON should look like:

```json
[
  {
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI that enables computers...",
      "There are three main types of machine learning: supervised...",
      "Machine learning algorithms use statistical techniques..."
    ],
    "answer": "Machine learning is a subset of artificial intelligence...",
    "reranking_scores": [0.95, 0.7, 0.5]  // Optional
  },
  ...
]
```

## Configuration Options

### Retrieval Models

```bash
# Default: SPLADE-v3 (paper's approach)
--retriever_model naver/splade-cocondenser-ensembledistil

# Alternative: Dense retrieval
# (requires modifying retrieval.py to support dense encoders)
```

### Reranking Models

```bash
# Lightweight (default)
--reranker_model cross-encoder/ms-marco-MiniLM-L-6-v2

# DeBERTa-v3 (closer to paper, slower)
--reranker_model cross-encoder/ms-marco-deberta-v3-base

# Large DeBERTa-v3 (best quality)
--reranker_model cross-encoder/ms-marco-deberta-v3-large
```

### Teacher Models

```bash
# Default: Mistral-7B-Instruct (paper's approach)
--teacher_model mistralai/Mistral-7B-Instruct-v0.2

# Alternative: Llama-3 (more powerful)
--teacher_model meta-llama/Meta-Llama-3-8B-Instruct

# Smaller for testing
--teacher_model microsoft/phi-2
```

## Paper's Dataset Details

From OSCAR paper Section 4:

### Queries (893k total)
- **MS MARCO dev**: 6,980 queries
- **PISCO**: 886k+ queries from Wikipedia

### Documents
- **Corpus**: Wikipedia-KILT (5.9M articles)
- **Retrieval**: SPLADE-v3
- **Top-K**: Retrieve 20, use 5 after reranking (for some experiments)

### Teacher
- **Model**: Mistral-7B-Instruct-v0.2
- **Task**: Generate answers given query + documents
- **Purpose**: Student learns to compress documents while maintaining generation quality

### Reranking (optional)
- **Model**: DeBERTa-v3 cross-encoder
- **Purpose**: Joint compression + reranking with RR token
- **Loss weight**: λ = 0.05

## Advanced Usage

### Parallel Processing

For large-scale dataset creation, parallelize across multiple GPUs:

```bash
# Split queries into chunks
split -l 1000 queries.txt queries_chunk_

# Process each chunk on a different GPU
for chunk in queries_chunk_*; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m scaledown.data.prepare_dataset \
        --queries_file $chunk \
        --corpus_path kilt_knowledgesource.json \
        --output_file train_data_${chunk}.json &
done

# Merge results
python -c "
import json
import glob

all_data = []
for f in glob.glob('train_data_*.json'):
    with open(f) as fin:
        all_data.extend(json.load(fin))

with open('train_data_full.json', 'w') as fout:
    json.dump(all_data, fout, indent=2)
"
```

### Custom Retrieval

If you already have retrieved documents:

```python
from scaledown.data.teacher import generate_teacher_answers
from scaledown.data.reranker import add_reranking_scores

# Your pre-retrieved data
queries_with_docs = [
    {
        "query": "What is ...",
        "documents": [
            {"title": "...", "text": "..."},
            {"title": "...", "text": "..."},
        ]
    },
    ...
]

# Add reranking scores (optional)
queries_with_docs = add_reranking_scores(queries_with_docs)

# Generate teacher answers
training_data = generate_teacher_answers(queries_with_docs)

# Save
import json
with open("train_data.json", "w") as f:
    json.dump(training_data, f, indent=2)
```

### Using Different Corpora

Instead of Wikipedia-KILT, use your own corpus:

```python
from scaledown.data.retrieval import SPLADERetriever

# Your custom corpus
corpus = [
    {"id": "1", "title": "...", "text": "..."},
    {"id": "2", "title": "...", "text": "..."},
    ...
]

# Retrieve
retriever = SPLADERetriever()
results = retriever.retrieve(queries, corpus, top_k=5)
```

## Mock Dataset for Testing

For quick testing without downloading Wikipedia-KILT:

```python
import json

# Create minimal mock data
mock_data = [
    {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI.",
            "ML algorithms learn from data.",
            "There are supervised and unsupervised methods."
        ],
        "answer": "Machine learning is a subset of AI that learns from data."
    },
    # ... more examples
]

with open("mock_data.json", "w") as f:
    json.dump(mock_data, f, indent=2)

# Use for training
# python train.py --train_data mock_data.json
```

## Comparison: Dataset Approaches

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **OSCAR's approach** | Faithful to paper, proven results | Requires Wikipedia-KILT (35GB), slow | Reproducing paper results |
| **Existing QA datasets** | Easy, no retrieval needed | May not match RAG setting | Quick experiments |
| **Custom corpus** | Domain-specific | Requires your own documents | Production use |
| **Mock data** | Fast, no downloads | Not realistic | Testing code |

## Troubleshooting

### Out of Memory (Retrieval)

```bash
# Reduce corpus size
--max_corpus_size 100000

# Process in smaller batches (modify retrieval.py)
# Change batch_size=32 to batch_size=8
```

### Out of Memory (Teacher Generation)

```bash
# Use 8-bit quantization
--teacher_8bit

# Use smaller teacher model
--teacher_model microsoft/phi-2
```

### Slow Processing

```bash
# Reduce top-K retrieval
--top_k_retrieval 5

# Skip reranking
# (don't use --enable_reranking or --top_k_reranking)

# Use smaller teacher model
--teacher_model microsoft/phi-2
```

### Missing Dependencies

```bash
# Install dataset generation dependencies
pip install sentence-transformers requests bitsandbytes
```

## Next Steps

After generating your dataset:

1. **Train ScaleDown model**:
   ```bash
   python train.py --train_data train_data.json
   ```

2. **Evaluate** on your RAG benchmark

3. **Compare** N-Layers vs ModernBERT compressor performance

4. **Experiment** with different compression rates and configurations

## References

- **OSCAR paper**: https://arxiv.org/abs/2504.07109
- **Wikipedia-KILT**: https://github.com/facebookresearch/KILT
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **SPLADE**: https://github.com/naver/splade
- **DeBERTa-v3**: https://huggingface.co/cross-encoder/ms-marco-deberta-v3-base
