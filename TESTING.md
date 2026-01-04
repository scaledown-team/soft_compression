# ScaleDown Testing Guide

This guide explains how to test your ScaleDown installation and verify that training works correctly before running full-scale experiments.

## Quick Test

### Option 1: Automated Test Script (Recommended)

The easiest way to verify everything works:

```bash
# Test N-Layers compressor (OSCAR paper variant)
python test_training.py --compressor_type n_layers

# Test ModernBERT compressor (novel variant)
python test_training.py --compressor_type modernbert

# Test both variants
python test_training.py --test_both
```

This will run 4 tests:
1. ✓ Model initialization
2. ✓ Dataset creation
3. ✓ Forward pass
4. ✓ Training loop (5 steps)

**Expected output:**
```
==================================================================================
TEST SUMMARY
==================================================================================
  ✓ PASS: Model Initialization
  ✓ PASS: Dataset Creation
  ✓ PASS: Forward Pass
  ✓ PASS: Training Loop
==================================================================================

🎉 All tests passed! Your ScaleDown setup is working correctly.
```

### Option 2: Manual Quick Test

Run a minimal training session manually:

```python
import torch
from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset
from scaledown.training import ScaleDownTrainer

# 1. Create minimal test data
test_data = [
    {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI.",
            "ML algorithms learn from data.",
        ],
        "answer": "Machine learning is a subset of AI that learns from data."
    },
    # ... add 9 more similar examples
]

# 2. Configure for quick test
config = ScaleDownConfig(
    compressor_type="n_layers",
    num_compressor_layers=5,      # Fewer layers = faster
    num_memory_tokens=4,           # Smaller compression
    compression_rate=8,
    batch_size=2,                  # Small batch
    num_epochs=1,
    max_steps=10,                  # Just 10 steps
    device_type="gpu" if torch.cuda.is_available() else "cpu",
)

# 3. Create dataset and model
dataset = ScaleDownDataset(test_data, config)
model = ScaleDownModel(config)

# 4. Train for 10 steps
trainer = ScaleDownTrainer(model, config, dataset, output_dir="./test_output")
trainer.train()

print("✓ Test training completed successfully!")
```

## Step-by-Step Verification

### Step 1: Verify Installation

```bash
# Check Python environment
python --version  # Should be 3.8+

# Check installed packages
pip list | grep torch
pip list | grep transformers
pip list | grep peft

# Test imports
python -c "from scaledown import ScaleDownConfig, ScaleDownModel; print('✓ ScaleDown imports work')"
```

### Step 2: Check GPU/Hardware

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Minimum requirements:**
- **GPU**: 16GB+ VRAM (24GB+ recommended for N-Layers with 8 layers)
- **CPU**: Works but very slow (not recommended)
- **Trainium**: Trn1.2xlarge or larger

### Step 3: Test Model Components

#### Test Compressor (N-Layers)

```python
from scaledown import ScaleDownConfig
from scaledown.models import NLayersCompressor

config = ScaleDownConfig(compressor_type="n_layers", num_compressor_layers=5)

# Initialize compressor
# Note: NLayersCompressor requires the generator model
from scaledown import ScaleDownModel
model = ScaleDownModel(config)

print(f"✓ N-Layers compressor initialized")
print(f"  Layers: {config.num_compressor_layers}")
print(f"  Memory tokens: {config.num_memory_tokens}")
```

#### Test Compressor (ModernBERT)

```python
from scaledown import ScaleDownConfig
from scaledown.models import ModernBERTCompressor

config = ScaleDownConfig(compressor_type="modernbert")
compressor = ModernBERTCompressor(config)

print(f"✓ ModernBERT compressor initialized")
print(f"  Model: {config.modernbert_model_name}")
print(f"  Memory tokens: {config.num_memory_tokens}")
```

#### Test Generator

```python
from scaledown import ScaleDownConfig
from scaledown.models import ScaleDownGenerator

config = ScaleDownConfig()
generator = ScaleDownGenerator(config)

print(f"✓ Generator initialized")
print(f"  Base model: {config.generator_model_name}")
print(f"  LoRA enabled: {config.use_lora}")
```

### Step 4: Test Dataset Processing

```python
from scaledown import ScaleDownConfig
from scaledown.data import ScaleDownDataset

# Create test data
data = [
    {
        "query": "What is X?",
        "documents": ["X is Y.", "X relates to Z."],
        "answer": "X is Y and relates to Z."
    }
]

config = ScaleDownConfig()
dataset = ScaleDownDataset(data, config)

# Test data loading
sample = dataset[0]
print(f"✓ Dataset created")
print(f"  Query tokens: {sample['query_input_ids'].shape}")
print(f"  Doc tokens: {sample['doc_input_ids'].shape}")
print(f"  Answer tokens: {sample['answer_input_ids'].shape}")
```

### Step 5: Test Training Step

```python
from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset, collate_fn
from torch.utils.data import DataLoader
import torch

# Setup
config = ScaleDownConfig(batch_size=2, num_compressor_layers=5)
data = [
    {
        "query": f"Query {i}",
        "documents": [f"Doc 1 for query {i}", f"Doc 2 for query {i}"],
        "answer": f"Answer {i}"
    }
    for i in range(4)
]

# Create dataset and dataloader
dataset = ScaleDownDataset(data, config)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Create model
model = ScaleDownModel(config)
model.train()

# Get batch and run forward pass
batch = next(iter(dataloader))
outputs = model(
    query_input_ids=batch['query_input_ids'],
    query_attention_mask=batch['query_attention_mask'],
    doc_input_ids=batch['doc_input_ids'],
    doc_attention_mask=batch['doc_attention_mask'],
    answer_input_ids=batch['answer_input_ids'],
    answer_attention_mask=batch['answer_attention_mask'],
)

print(f"✓ Training step successful")
print(f"  Loss: {outputs.loss.item():.4f}")

# Test backward pass
outputs.loss.backward()
print(f"✓ Backward pass successful")
```

## Performance Benchmarks

Expected performance for quick tests on common hardware:

### N-Layers Compressor (5 layers)

| Hardware | Batch Size | Speed (steps/sec) | Memory |
|----------|-----------|-------------------|---------|
| RTX 4090 (24GB) | 4 | ~2.5 | ~18GB |
| A100 (40GB) | 8 | ~4.0 | ~30GB |
| V100 (16GB) | 2 | ~1.5 | ~14GB |
| CPU (32 cores) | 1 | ~0.05 | ~12GB |

### ModernBERT Compressor

| Hardware | Batch Size | Speed (steps/sec) | Memory |
|----------|-----------|-------------------|---------|
| RTX 4090 (24GB) | 8 | ~4.0 | ~12GB |
| A100 (40GB) | 16 | ~6.0 | ~20GB |
| V100 (16GB) | 4 | ~2.5 | ~10GB |
| CPU (32 cores) | 1 | ~0.1 | ~8GB |

**ModernBERT is ~2× faster** due to smaller model size (149M vs 1.9B params).

## Common Issues and Solutions

### Issue 1: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
config = ScaleDownConfig(batch_size=1)

# 2. Use fewer compressor layers (N-Layers only)
config = ScaleDownConfig(num_compressor_layers=5)  # Instead of 8

# 3. Enable gradient checkpointing (already default)
config = ScaleDownConfig(gradient_checkpointing=True)

# 4. Use BF16 (already default)
config = ScaleDownConfig(use_bf16=True)

# 5. Reduce memory tokens
config = ScaleDownConfig(num_memory_tokens=4)  # Instead of 8
```

### Issue 2: Slow Training

**Symptoms:**
Training is much slower than benchmarks above.

**Solutions:**
```python
# 1. Use ModernBERT instead of N-Layers
config = ScaleDownConfig(compressor_type="modernbert")

# 2. Use fewer compressor layers
config = ScaleDownConfig(num_compressor_layers=5)

# 3. Reduce sequence lengths
config = ScaleDownConfig(
    max_query_length=64,   # Instead of 128
    max_doc_length=64,     # Instead of 128
)

# 4. Check GPU utilization
# Run: nvidia-smi -l 1
# Should see ~95%+ GPU utilization
```

### Issue 3: NaN Loss

**Symptoms:**
```
Loss: nan
```

**Solutions:**
```python
# 1. Reduce learning rate
config = ScaleDownConfig(
    learning_rate_generator=5e-5,           # Instead of 1e-4
    learning_rate_compressor_nlayers=1e-5,  # Instead of 5e-5
)

# 2. Enable gradient clipping (already default)
config = ScaleDownConfig(max_grad_norm=1.0)

# 3. Check data quality
# Make sure documents and answers are not empty
for item in data:
    assert item['query'].strip(), "Empty query"
    assert all(d.strip() for d in item['documents']), "Empty document"
    assert item['answer'].strip(), "Empty answer"
```

### Issue 4: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'scaledown'
```

**Solutions:**
```bash
# 1. Install in development mode
pip install -e .

# 2. Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# 3. Make sure you're in the right directory
cd /path/to/soft_compression
python -c "from scaledown import ScaleDownConfig; print('✓ Works')"
```

### Issue 5: Model Loading Errors

**Symptoms:**
```
OSError: Can't load tokenizer for 'mistralai/Mistral-7B-Instruct-v0.2'
```

**Solutions:**
```bash
# 1. Check internet connection
curl https://huggingface.co

# 2. Login to HuggingFace (for gated models)
huggingface-cli login

# 3. Use a different model
python -c "
from scaledown import ScaleDownConfig
config = ScaleDownConfig(generator_model_name='gpt2')  # Smaller, always available
"
```

## Full Training Test (30 minutes)

Once quick tests pass, run a longer test to verify stability:

```bash
# 1. Create test dataset (100 examples)
python -m scaledown.data.prepare_dataset \
  --num_synthetic_queries 100 \
  --output_file test_100.json

# 2. Train for 1 epoch
python train.py \
  --compressor_type n_layers \
  --num_layers 5 \
  --train_data test_100.json \
  --batch_size 4 \
  --num_epochs 1 \
  --output_dir ./test_training_output \
  --logging_steps 10

# 3. Check results
# Loss should decrease over the epoch
# Final checkpoint should be saved to ./test_training_output/final/
```

**Expected behavior:**
- Training starts without errors
- Loss decreases over time (may fluctuate)
- GPU memory usage is stable
- No OOM errors
- Checkpoint saved successfully

## Next Steps

Once all tests pass:

1. **Generate real training data**:
   ```bash
   # See DATASET_PREPARATION.md
   python -m scaledown.data.prepare_dataset \
     --download_ms_marco \
     --corpus_path kilt_knowledgesource.json \
     --output_file train_data.json
   ```

2. **Run full training**:
   ```bash
   python train.py \
     --train_data train_data.json \
     --compressor_type n_layers \
     --num_layers 8 \
     --batch_size 128 \
     --num_epochs 1 \
     --output_dir ./checkpoints
   ```

3. **Evaluate on your RAG benchmark**

4. **Compare N-Layers vs ModernBERT performance**

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model Parameters

```python
from scaledown import ScaleDownModel, ScaleDownConfig

config = ScaleDownConfig(compressor_type="n_layers")
model = ScaleDownModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Check which parameters are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel():,}")
```

### Profile Training

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Run one training step
    outputs = model(...)
    outputs.loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Support

If tests fail:
1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Check [GitHub Issues](https://github.com/yourusername/scaledown/issues)
4. Provide full error logs when reporting issues

---

**Remember:** The test script is designed to catch issues early. If `test_training.py` passes, you're ready for full-scale training!
