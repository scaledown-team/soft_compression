# ScaleDown Installation Guide

## No Package Installation Needed!

This is research code - you can use it directly without installing as a package.

## Quick Setup (2 minutes)

### Option 1: Automatic Setup (Linux/Mac)

```bash
# Navigate to project
cd /path/to/soft_compression

# Run setup script
bash setup_env.sh
```

This installs all dependencies and adds the directory to your Python path.

### Option 2: Manual Setup (All Platforms)

```bash
# Navigate to project
cd /path/to/soft_compression

# Install dependencies
pip install torch>=2.0.0 transformers>=4.40.0 peft>=0.10.0 accelerate>=0.27.0
pip install datasets>=2.14.0 tokenizers>=0.15.0
pip install tqdm>=4.65.0 numpy>=1.24.0 matplotlib>=3.7.0

# For dataset generation (optional)
pip install sentence-transformers>=2.3.0 requests>=2.31.0 bitsandbytes>=0.42.0

# For experiment tracking (optional)
pip install wandb>=0.16.0
```

That's it! No `pip install -e .` needed.

## Usage

All scripts automatically add the correct path. Just run them:

```bash
# Test the setup
python test_training.py --test_both

# Prepare real data
python prepare_small_real_dataset.py --dataset squad --num_examples 500

# Train with evaluation
python train_with_evaluation.py --train_data small_real_dataset.json
```

## How It Works

Each script includes this at the top:

```python
# Add current directory to path (no package installation needed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Now you can import
from scaledown import ScaleDownConfig, ScaleDownModel
```

This means:
- ✅ No `pip install -e .` required
- ✅ No setup.py needed
- ✅ No package building
- ✅ Just install dependencies and run

## Directory Structure

```
soft_compression/
├── scaledown/              # Source code (NOT installed as package)
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   ├── training/
│   ├── data/
│   └── evaluation/
├── test_training.py        # Scripts that use scaledown modules
├── train_with_evaluation.py
├── prepare_small_real_dataset.py
└── requirements.txt        # Just dependencies, no setup.py
```

## Using in Your Own Scripts

If you create new scripts, add this at the top:

```python
# your_script.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scaledown import ScaleDownConfig, ScaleDownModel
# ... rest of your code
```

Or import the setup helper:

```python
# your_script.py
import setup_path  # Adds path automatically

from scaledown import ScaleDownConfig, ScaleDownModel
# ... rest of your code
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'scaledown'"

**Solution 1:** Make sure you're running from the `soft_compression/` directory:
```bash
cd /path/to/soft_compression
python test_training.py
```

**Solution 2:** Manually add to path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/soft_compression"
python test_training.py
```

**Solution 3:** Use the setup script:
```python
import setup_path  # At top of your script
```

### "ImportError: cannot import name..."

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Still having issues?

The old way (if you really want to install as a package):
```bash
pip install -e .
```

But this is **not recommended** for research code. Better to use the direct import approach above.

## For Colab

In Colab, the scripts work automatically after cloning:

```python
# Clone repo
!git clone https://github.com/yourusername/scaledown.git
%cd scaledown/soft_compression

# Install dependencies
!pip install -q torch transformers peft accelerate datasets tqdm matplotlib

# Run directly (no installation)
!python test_training.py --test_both
```

## Why No Package Installation?

Research code benefits from:
1. **Easy editing** - Modify `scaledown/` files and see changes immediately
2. **No rebuild** - No need to reinstall after each change
3. **Flexibility** - Copy scripts anywhere, they work
4. **Simplicity** - One less step for users
5. **Standard practice** - Many research repos work this way

## Comparison

| Approach | Steps | Pros | Cons |
|----------|-------|------|------|
| **Direct use (current)** | Install deps, run | Simple, editable | Need to manage path |
| **Package install** | Install deps, pip install -e . | Python path handled | Extra step, need to reinstall after edits |

For research: **Direct use is better** ✅

For production: Package installation makes sense.

## Next Steps

Setup is done! Now:

1. **Test**: `python test_training.py --test_both`
2. **Get data**: `python prepare_small_real_dataset.py --dataset squad --num_examples 500`
3. **Train**: `python train_with_evaluation.py --train_data small_real_dataset.json`

See [REAL_DATA_TRAINING.md](./REAL_DATA_TRAINING.md) for details.
