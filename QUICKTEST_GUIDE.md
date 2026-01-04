# Quick Test Guide - Start Here!

This is a **5-minute guide** to verify your ScaleDown installation works before running full training.

## TL;DR - Run This

```bash
# Navigate to the project
cd /Users/soham/Desktop/scaledown-project/soft_compression

# Install the package
pip install -e .

# Run the test
python test_training.py --test_both
```

**Expected output:** All 4 tests pass ✓

---

## What Gets Tested

The test script verifies 4 critical components:

### 1. Model Initialization ✓
- Creates N-Layers and ModernBERT compressors
- Initializes generator with LoRA
- Verifies all components are present

### 2. Dataset Creation ✓
- Processes synthetic training examples
- Tokenizes queries, documents, and answers
- Inserts memory tokens correctly

### 3. Forward Pass ✓
- Runs data through compressor → generator
- Computes loss
- Validates output format

### 4. Training Loop ✓
- Runs 5 training steps
- Performs backward pass
- Updates model parameters
- Checks for loss decrease

---

## Test Output Explained

### Success Output

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

**This means:** You're ready to generate real data and start training!

### Failure Output

```
==================================================================================
TEST SUMMARY
==================================================================================
  ✓ PASS: Model Initialization
  ✓ PASS: Dataset Creation
  ✗ FAIL: Forward Pass
  ...
==================================================================================

❌ Some tests failed. Please check the errors above.
```

**This means:** Something is wrong. Check the error message and see [TESTING.md](./TESTING.md) for solutions.

---

## Common Issues (Quick Fixes)

### Issue: `ModuleNotFoundError: No module named 'scaledown'`

**Fix:**
```bash
pip install -e .
```

### Issue: `CUDA out of memory`

**Fix:** Your GPU doesn't have enough memory. This is expected if you have <16GB VRAM. The test will still work on CPU (just slower).

Or reduce memory usage:
```bash
# Test with smaller config (edit test_training.py):
config = ScaleDownConfig(
    num_compressor_layers=5,  # Fewer layers
    num_memory_tokens=4,      # Smaller compression
    batch_size=1,             # Smaller batch
)
```

### Issue: Test runs but is very slow

**Normal behavior on CPU.** Expected speeds:
- **GPU (RTX 4090):** ~30 seconds for full test
- **CPU (32 cores):** ~5-10 minutes for full test

### Issue: `Can't load tokenizer for 'mistralai/Mistral-7B-Instruct-v0.2'`

**Fix:** Check internet connection or login to HuggingFace:
```bash
huggingface-cli login
```

---

## Test Options

```bash
# Test only N-Layers compressor (OSCAR paper variant)
python test_training.py --compressor_type n_layers

# Test only ModernBERT compressor (novel variant)
python test_training.py --compressor_type modernbert

# Test both (recommended)
python test_training.py --test_both

# Test with more examples (slower but more thorough)
python test_training.py --test_both --num_examples 50
```

---

## What Happens During the Test

1. **Downloads models** (~15GB first time):
   - Mistral-7B-Instruct-v0.2 (generator)
   - ModernBERT-base (if testing ModernBERT)
   - This only happens once - subsequent runs are fast

2. **Creates synthetic data**:
   - 10 fake query-document-answer triples
   - No need for real data

3. **Runs 5 training steps**:
   - Compresses documents
   - Generates answers
   - Computes loss
   - Updates weights

4. **Reports results**:
   - Pass/fail for each test
   - Performance metrics

**Total time:** 30 seconds - 2 minutes (GPU) or 5-10 minutes (CPU)

---

## Next Steps After Tests Pass

### Option 1: Quick Training Test (30 minutes)

Generate 100 examples and train for real:

```bash
# 1. Create small test dataset (no downloads needed)
python example_dataset_generation.py  # Creates synthetic_train_data.json

# 2. Train for 1 epoch
python train.py \
  --train_data synthetic_train_data.json \
  --compressor_type n_layers \
  --num_layers 5 \
  --batch_size 4 \
  --num_epochs 1 \
  --output_dir ./quick_test_output
```

### Option 2: Generate Real Data (Following OSCAR Paper)

See [DATASET_PREPARATION.md](./DATASET_PREPARATION.md) for the full pipeline:

```bash
# Download Wikipedia-KILT (35GB)
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json

# Generate training data with retrieval + teacher LLM
python -m scaledown.data.prepare_dataset \
  --download_ms_marco \
  --corpus_path kilt_knowledgesource.json \
  --output_file train_data.json \
  --enable_reranking
```

### Option 3: Use Existing QA Dataset

Skip retrieval and use existing data:

```python
from datasets import load_dataset

# Load SQuAD
squad = load_dataset("squad", split="train")

# Convert to ScaleDown format
data = []
for ex in squad[:1000]:  # First 1000 examples
    data.append({
        "query": ex["question"],
        "documents": [ex["context"]],  # Single document
        "answer": ex["answers"]["text"][0],
    })

# Save
import json
with open("squad_data.json", "w") as f:
    json.dump(data, f)

# Train
# python train.py --train_data squad_data.json
```

---

## Understanding Test Performance

### N-Layers Compressor (5 layers)

- **Parameters:** ~1.5B (compressor) + 7B (generator with LoRA)
- **Memory:** ~12-18GB GPU RAM
- **Speed:** ~2-3 steps/sec (RTX 4090)

### ModernBERT Compressor

- **Parameters:** 149M (compressor) + 7B (generator with LoRA)
- **Memory:** ~8-12GB GPU RAM
- **Speed:** ~4-5 steps/sec (RTX 4090)
- **Note:** 2× faster than N-Layers!

---

## Troubleshooting Decision Tree

```
Tests failed?
├─ Import error → Run: pip install -e .
├─ CUDA OOM → Reduce batch_size or use ModernBERT
├─ Very slow → Normal on CPU, use GPU if possible
├─ NaN loss → Check data, reduce learning rate
└─ Other error → See TESTING.md for detailed solutions
```

---

## Full Documentation

- **[TESTING.md](./TESTING.md)** - Comprehensive testing guide with all solutions
- **[README.md](./README.md)** - Project overview and API reference
- **[DATASET_PREPARATION.md](./DATASET_PREPARATION.md)** - How to create training data
- **[QUICKSTART.md](./QUICKSTART.md)** - Minimal working examples
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Technical details

---

## Get Help

If tests fail after trying the quick fixes above:

1. Check [TESTING.md](./TESTING.md) for detailed troubleshooting
2. Read the full error message carefully
3. Verify GPU setup: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check dependencies: `pip list | grep -E "torch|transformers|peft"`

---

**Remember:** The test script is designed to catch issues early. If it passes, you're ready to train ScaleDown! 🚀
