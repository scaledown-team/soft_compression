#!/bin/bash

# Setup script for ScaleDown research code
# Just installs dependencies, no package installation needed

echo "=================================="
echo "ScaleDown Environment Setup"
echo "=================================="
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install torch>=2.0.0 transformers>=4.40.0 peft>=0.10.0 accelerate>=0.27.0

# Install data processing
echo ""
echo "Installing data processing dependencies..."
pip install datasets>=2.14.0 tokenizers>=0.15.0

# Install utilities
echo ""
echo "Installing utilities..."
pip install tqdm>=4.65.0 numpy>=1.24.0 matplotlib>=3.7.0

# Install dataset generation dependencies (optional)
echo ""
echo "Installing dataset generation dependencies..."
pip install sentence-transformers>=2.3.0 requests>=2.31.0 bitsandbytes>=0.42.0

# Optional: wandb for logging
read -p "Install wandb for experiment tracking? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install wandb>=0.16.0
fi

# Add current directory to PYTHONPATH
echo ""
echo "Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" >> ~/.bashrc

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To use ScaleDown in this session:"
echo "  export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\""
echo ""
echo "This has been added to your ~/.bashrc for future sessions."
echo ""
echo "Next steps:"
echo "  1. Test: python test_training.py --test_both"
echo "  2. Train: python prepare_small_real_dataset.py --dataset squad --num_examples 500"
echo "           python train_with_evaluation.py --train_data small_real_dataset.json"
echo "=================================="
