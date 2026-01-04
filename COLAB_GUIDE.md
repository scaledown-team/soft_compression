# Running ScaleDown in Google Colab

The easiest way to try ScaleDown without local setup!

## 🚀 Quick Start (One Click)

1. **Open the notebook**: [ScaleDown_Colab.ipynb](./ScaleDown_Colab.ipynb)
2. **Click "Open in Colab"** badge at the top
3. **Select GPU runtime**: Runtime → Change runtime type → GPU (T4)
4. **Run all cells**: Runtime → Run all

That's it! The notebook will:
- ✅ Install ScaleDown
- ✅ Run tests
- ✅ Train a demo model (5 minutes)

## 📋 Step-by-Step Instructions

### Step 1: Open Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Upload notebook
3. Select `ScaleDown_Colab.ipynb` from this repo

**Option B: From GitHub** (if you've pushed to GitHub)
1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub tab
3. Enter: `yourusername/scaledown`
4. Select `ScaleDown_Colab.ipynb`

### Step 2: Enable GPU

**IMPORTANT**: Make sure you're using GPU!

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type:
   - **T4** (Free tier) - Good for demos and small experiments
   - **A100** (Colab Pro) - Best for full training

### Step 3: Run the Notebook

**Quick Demo (5 minutes):**
- Run cells under "Option A: Quick Demo"
- Uses 3 synthetic examples
- Trains for 10 steps
- Perfect for testing

**Small Training (30 minutes):**
- Run cells under "Option B: Train with Small Dataset"
- Generates 100 examples
- Trains for 1 epoch
- Good for learning

**Full Training (hours):**
- Run cells under "Option C: Full Training"
- Downloads Wikipedia-KILT (35GB)
- Uses real MS MARCO queries
- Requires Colab Pro with A100

## 💡 Colab-Specific Tips

### 1. GPU Memory Limits

**Free Tier (T4 - 16GB):**
```python
config = ScaleDownConfig(
    compressor_type="modernbert",  # Use ModernBERT (2× faster)
    batch_size=1,                  # Small batch
    num_memory_tokens=4,           # Reduce compression size
)
```

**Colab Pro (A100 - 40GB):**
```python
config = ScaleDownConfig(
    compressor_type="n_layers",    # Can use N-Layers
    num_compressor_layers=8,
    batch_size=8,                  # Larger batch
    num_memory_tokens=8,
)
```

### 2. Session Timeouts

Colab sessions disconnect after:
- **Free**: 12 hours max, idle timeout ~90 minutes
- **Pro**: 24 hours max, idle timeout ~6 hours

**Solution**: Save checkpoints frequently

```python
config = ScaleDownConfig(
    save_steps=100,  # Save every 100 steps
    output_dir="/content/drive/MyDrive/scaledown_checkpoints",  # Save to Drive
)
```

### 3. Mount Google Drive

Save results to Drive so they persist:

```python
from google.colab import drive
drive.mount('/content/drive')

# Train with output to Drive
!python train.py \
  --train_data data.json \
  --output_dir /content/drive/MyDrive/scaledown_output
```

### 4. Download Results

After training:

```python
# Download checkpoint
from google.colab import files
files.download('./demo_output/final/pytorch_model.bin')

# Or copy to Drive
!cp -r ./demo_output /content/drive/MyDrive/
```

### 5. Monitor GPU Usage

```python
# Check GPU memory
!nvidia-smi

# In Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

### 6. Clear Memory Between Runs

If you get OOM errors:

```python
import torch
import gc

# Delete models
if 'model' in locals():
    del model
if 'trainer' in locals():
    del trainer

# Clear cache
gc.collect()
torch.cuda.empty_cache()
```

## 🎯 Recommended Workflow for Colab

### For Free Tier (T4 GPU)

```bash
# 1. Quick test (5 min)
# Run "Option A: Quick Demo" cells

# 2. Generate small dataset (5 min)
!python example_dataset_generation.py

# 3. Train (30 min)
!python train.py \
  --train_data synthetic_train_data.json \
  --compressor_type modernbert \
  --batch_size 2 \
  --num_epochs 1
```

### For Colab Pro (A100 GPU)

```bash
# 1. Download small corpus subset (10 min)
!wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json

# 2. Generate dataset (1 hour)
!python -m scaledown.data.prepare_dataset \
  --num_synthetic_queries 1000 \
  --corpus_path kilt_knowledgesource.json \
  --max_corpus_size 50000 \
  --output_file train_data.json \
  --teacher_8bit

# 3. Train (2-4 hours)
!python train.py \
  --train_data train_data.json \
  --compressor_type n_layers \
  --num_layers 8 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir /content/drive/MyDrive/scaledown_output
```

## 📊 Expected Performance

### Training Speed (steps/second)

| Hardware | N-Layers | ModernBERT |
|----------|----------|------------|
| T4 (16GB) | ~0.8 | ~2.0 |
| A100 (40GB) | ~4.0 | ~6.0 |

### Memory Usage

| Config | N-Layers (8 layers) | ModernBERT |
|--------|---------------------|------------|
| Batch size 1 | ~14GB | ~8GB |
| Batch size 4 | ~20GB+ (OOM on T4) | ~12GB |
| Batch size 8 | N/A (needs A100) | ~18GB |

## ⚠️ Common Colab Issues

### Issue 1: "Runtime disconnected"

**Cause**: Idle timeout or memory limit exceeded

**Solutions**:
1. Keep Colab tab active
2. Reduce batch size
3. Enable checkpointing
4. Use Colab Pro for longer sessions

### Issue 2: "CUDA out of memory"

**Solutions**:
```python
# Use smaller config
config = ScaleDownConfig(
    compressor_type="modernbert",  # Smaller model
    batch_size=1,                  # Reduce batch
    num_memory_tokens=4,           # Less compression
)

# Or clear memory
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue 3: "Disk quota exceeded"

**Cause**: Wikipedia-KILT (35GB) fills up Colab disk

**Solutions**:
1. Use `max_corpus_size` to limit corpus
2. Stream data instead of loading all at once
3. Save intermediate results to Drive

```bash
# Use small corpus subset
!python -m scaledown.data.prepare_dataset \
  --max_corpus_size 10000 \
  ...
```

### Issue 4: Slow downloads

**Cause**: Downloading models from HuggingFace

**Solutions**:
1. First run will be slow (downloads 15GB+ of models)
2. Subsequent runs reuse cached models
3. Models cached in `/root/.cache/huggingface/`

### Issue 5: "Cannot find module scaledown"

**Solution**:
```python
# Make sure you ran the installation cell
!pip install -e .

# Or run from correct directory
%cd /content/scaledown/soft_compression
```

## 🔧 Optimization for Colab

### Use 8-bit Quantization

Reduce memory usage:

```python
config = ScaleDownConfig(
    use_bf16=True,              # Already default
    gradient_checkpointing=True, # Already default
)

# For teacher LLM generation
!python -m scaledown.data.prepare_dataset \
  --teacher_8bit \
  ...
```

### Enable Mixed Precision

Already enabled by default in ScaleDown, but you can verify:

```python
from scaledown.training import ScaleDownTrainer

trainer = ScaleDownTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    # Mixed precision enabled automatically
)
```

### Reduce Sequence Lengths

For faster training on limited hardware:

```python
config = ScaleDownConfig(
    max_query_length=64,   # Default: 128
    max_doc_length=64,     # Default: 128
    max_answer_length=64,  # Default: 128
)
```

## 📦 Save and Load Models

### Save Model

```python
# During training (automatic)
trainer = ScaleDownTrainer(
    ...,
    output_dir="/content/drive/MyDrive/scaledown",
    save_steps=500,
)

# Manual save
model.save_pretrained("/content/drive/MyDrive/scaledown/final")
```

### Load Model

```python
from scaledown import ScaleDownConfig, ScaleDownModel
import torch

config = ScaleDownConfig(compressor_type="modernbert")
model = ScaleDownModel(config)

# Load weights
checkpoint_path = "/content/drive/MyDrive/scaledown/final/pytorch_model.bin"
model.load_state_dict(torch.load(checkpoint_path))
```

## 🎓 Learning Path

**Day 1: Quick Demo (30 min)**
1. Open Colab notebook
2. Run "Option A: Quick Demo"
3. See training in action

**Day 2: Small Experiment (2 hours)**
1. Generate 100-example dataset
2. Train with both compressors
3. Compare N-Layers vs ModernBERT

**Day 3+: Full Training (with Colab Pro)**
1. Download Wikipedia-KILT
2. Generate MS MARCO dataset
3. Train with full pipeline
4. Evaluate on RAG benchmark

## 📚 Additional Resources

- **Notebook**: [ScaleDown_Colab.ipynb](./ScaleDown_Colab.ipynb)
- **Local Testing**: [QUICKTEST_GUIDE.md](./QUICKTEST_GUIDE.md)
- **Dataset Generation**: [DATASET_PREPARATION.md](./DATASET_PREPARATION.md)
- **Full Documentation**: [README.md](./README.md)

## 💰 Cost Comparison

| Option | GPU | Cost | Best For |
|--------|-----|------|----------|
| **Colab Free** | T4 (16GB) | Free | Demos, learning, small experiments |
| **Colab Pro** | A100 (40GB) | $10/month | Full training, experiments |
| **Colab Pro+** | A100 (40GB) | $50/month | Long training runs, priority access |
| **Local GPU** | Your GPU | One-time | Development, repeated experiments |
| **AWS/GCP** | Custom | Pay-per-use | Production, large-scale |

**Recommendation**: Start with **Colab Free** for demos, upgrade to **Pro** if you need longer training runs.

---

**Ready to start?** Open [ScaleDown_Colab.ipynb](./ScaleDown_Colab.ipynb) and click "Open in Colab"! 🚀
