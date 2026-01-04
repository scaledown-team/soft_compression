#!/bin/bash

# ScaleDown Quick Test Script
# Runs a comprehensive test to verify your installation is working

set -e  # Exit on error

echo "======================================================================"
echo "ScaleDown Quick Test"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Check if in correct directory
if [ ! -f "setup.py" ]; then
    echo "✗ Error: Must run from soft_compression/ directory"
    exit 1
fi

# Check if package is installed
echo "Checking if ScaleDown is installed..."
if python -c "from scaledown import ScaleDownConfig" 2>/dev/null; then
    echo "✓ ScaleDown package is installed"
else
    echo "⚠ ScaleDown not installed. Installing now..."
    pip install -e . > /dev/null 2>&1
    echo "✓ ScaleDown installed"
fi
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    gpu_memory=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')" 2>/dev/null)
    echo "✓ GPU available: $gpu_name ($gpu_memory GB)"
else
    echo "⚠ No GPU detected. Training will be slow on CPU."
fi
echo ""

# Run test script
echo "======================================================================"
echo "Running automated tests..."
echo "======================================================================"
echo ""

# Run with both compressors
python test_training.py --test_both --num_examples 10

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ All tests passed!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Generate training data (see DATASET_PREPARATION.md)"
    echo "  2. Train model: python train.py --train_data your_data.json"
    echo ""
    echo "Documentation:"
    echo "  - README.md - General overview and quick start"
    echo "  - TESTING.md - Detailed testing guide"
    echo "  - DATASET_PREPARATION.md - How to create training data"
    echo "  - ARCHITECTURE.md - Technical details"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Tests failed. Check errors above."
    echo "======================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  - See TESTING.md for common issues and solutions"
    echo "  - Check that all dependencies are installed: pip install -e ."
    echo "  - Verify GPU setup if using CUDA"
    echo "======================================================================"
    exit 1
fi
