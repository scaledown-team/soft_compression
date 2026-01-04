"""
Quick test script to verify ScaleDown training pipeline.

This script runs a minimal training loop to ensure everything works correctly
before running full-scale training.

Usage:
    # Test with N-Layers compressor (faithful to OSCAR paper)
    python test_training.py --compressor_type n_layers

    # Test with ModernBERT compressor (novel variant)
    python test_training.py --compressor_type modernbert

    # Test both variants
    python test_training.py --test_both
"""

# Add current directory to path (no package installation needed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import torch

from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data import ScaleDownDataset
from scaledown.training import ScaleDownTrainer


def create_minimal_test_data(num_examples: int = 10) -> list:
    """
    Create minimal synthetic test data.

    Args:
        num_examples: Number of examples to generate

    Returns:
        List of training examples
    """
    print(f"Creating {num_examples} synthetic test examples...")

    data = []
    for i in range(num_examples):
        example = {
            "query": f"What is concept {i}?",
            "documents": [
                f"Concept {i} is an important topic in the field of study.",
                f"The definition of concept {i} relates to understanding key principles.",
                f"Experts agree that concept {i} has significant implications.",
            ],
            "answer": f"Concept {i} is an important topic that relates to key principles and has significant implications.",
        }
        data.append(example)

    return data


def test_model_initialization(compressor_type: str = "n_layers"):
    """
    Test 1: Model initialization.

    Verifies that the model can be created without errors.
    """
    print("\n" + "=" * 80)
    print(f"TEST 1: Model Initialization ({compressor_type})")
    print("=" * 80)

    try:
        config = ScaleDownConfig(
            compressor_type=compressor_type,
            num_compressor_layers=5 if compressor_type == "n_layers" else None,
            num_memory_tokens=4,  # Small for testing
            compression_rate=8,   # 8× compression (32 tokens → 4 embeddings)
            batch_size=2,         # Minimal batch size
            device_type="gpu" if torch.cuda.is_available() else "cpu",
        )

        print(f"\nConfiguration:")
        print(f"  Compressor: {config.compressor_type}")
        print(f"  Memory tokens: {config.num_memory_tokens}")
        print(f"  Compression rate: {config.compression_rate}×")
        print(f"  Device: {config.device_type}")

        print("\nInitializing model...")
        model = ScaleDownModel(config)

        print("✓ Model initialized successfully")

        # Check model components
        assert model.compressor is not None, "Compressor not initialized"
        assert model.generator is not None, "Generator not initialized"
        print("✓ All model components present")

        return True, config

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False, None


def test_dataset_creation(config: ScaleDownConfig, num_examples: int = 10):
    """
    Test 2: Dataset creation.

    Verifies that the dataset can process examples correctly.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Dataset Creation")
    print("=" * 80)

    try:
        # Create test data
        data = create_minimal_test_data(num_examples)

        print("\nCreating dataset...")
        dataset = ScaleDownDataset(data, config)

        print(f"✓ Dataset created with {len(dataset)} examples")

        # Test data loading
        print("\nTesting data loading...")
        sample = dataset[0]

        # Check required fields
        required_fields = ['query_input_ids', 'doc_input_ids', 'labels']
        for field in required_fields:
            assert field in sample, f"Missing field: {field}"

        print("✓ Dataset returns correct format")
        print(f"  Query shape: {sample['query_input_ids'].shape}")
        print(f"  Doc shape: {sample['doc_input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")

        return True, dataset

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(config: ScaleDownConfig, dataset: ScaleDownDataset):
    """
    Test 3: Forward pass.

    Verifies that the model can process a batch without errors.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass")
    print("=" * 80)

    try:
        from torch.utils.data import DataLoader
        from scaledown.data import collate_fn

        # Create model
        model = ScaleDownModel(config)
        model.eval()  # Eval mode for testing

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn,
        )

        # Get a batch
        print("\nGetting batch from dataloader...")
        batch = next(iter(dataloader))

        print(f"✓ Batch created")
        print(f"  Batch size: {batch['query_input_ids'].shape[0]}")

        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(
                query_input_ids=batch['query_input_ids'],
                query_attention_mask=batch['query_attention_mask'],
                doc_input_ids=batch['doc_input_ids'],
                doc_attention_mask=batch['doc_attention_mask'],
                memory_token_positions=batch['memory_token_positions'],
                labels=batch['labels'],
            )

        print("✓ Forward pass successful")
        print(f"  Loss: {outputs['loss'].item():.4f}")

        # Check output
        assert outputs['loss'] is not None, "Loss not computed"
        assert not torch.isnan(outputs['loss']), "Loss is NaN"
        assert outputs['loss'] > 0, "Loss is not positive"

        print("✓ Loss is valid")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop(
    config: ScaleDownConfig,
    dataset: ScaleDownDataset,
    num_steps: int = 5,
):
    """
    Test 4: Training loop.

    Runs a minimal training loop to verify optimization works.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Training Loop")
    print("=" * 80)

    try:
        # Create model and trainer
        print("\nInitializing trainer...")
        model = ScaleDownModel(config)

        # Update config for minimal training
        test_config = ScaleDownConfig(
            **{k: v for k, v in config.__dict__.items()},
            num_epochs=1,
            max_steps=num_steps,
            batch_size=2,
            logging_steps=1,
            save_steps=None,  # Don't save checkpoints
        )

        trainer = ScaleDownTrainer(
            model=model,
            config=test_config,
            train_dataset=dataset,
            eval_dataset=None,  # No eval for quick test
            output_dir="./test_checkpoints",
        )

        print(f"✓ Trainer initialized")
        print(f"  Training for {num_steps} steps")

        # Run training
        print(f"\nRunning training loop...")
        initial_loss = None
        final_loss = None

        # Monkey-patch to capture losses
        original_train = trainer.train
        losses = []

        def train_wrapper():
            # Store training_step to capture losses
            original_step = trainer.training_step

            def step_wrapper(batch):
                loss = original_step(batch)
                losses.append(loss.item())
                return loss

            trainer.training_step = step_wrapper
            return original_train()

        train_wrapper()

        print("✓ Training loop completed")

        # Check losses
        if len(losses) > 0:
            initial_loss = losses[0]
            final_loss = losses[-1]

            print(f"\n  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")

            # Loss should generally decrease (not strict for 5 steps)
            if final_loss < initial_loss:
                print("✓ Loss decreased (model is learning)")
            else:
                print("⚠ Loss did not decrease (expected for very short training)")

        return True

    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(compressor_type: str = "n_layers", num_examples: int = 10):
    """
    Run all tests for a given compressor type.

    Args:
        compressor_type: "n_layers" or "modernbert"
        num_examples: Number of training examples

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 80)
    print(f"SCALEDOWN TRAINING TEST SUITE - {compressor_type.upper()}")
    print("=" * 80)

    results = []

    # Test 1: Model initialization
    success, config = test_model_initialization(compressor_type)
    results.append(("Model Initialization", success))
    if not success:
        return False

    # Test 2: Dataset creation
    success, dataset = test_dataset_creation(config, num_examples)
    results.append(("Dataset Creation", success))
    if not success:
        return False

    # Test 3: Forward pass
    success = test_forward_pass(config, dataset)
    results.append(("Forward Pass", success))
    if not success:
        return False

    # Test 4: Training loop
    success = test_training_loop(config, dataset, num_steps=5)
    results.append(("Training Loop", success))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All tests passed! Your ScaleDown setup is working correctly.")
        print(f"\nNext steps:")
        print(f"  1. Generate training data (see DATASET_PREPARATION.md)")
        print(f"  2. Run full training: python train.py --train_data your_data.json")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

    return all_passed


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test ScaleDown training pipeline"
    )

    parser.add_argument(
        "--compressor_type",
        type=str,
        choices=["n_layers", "modernbert"],
        default="n_layers",
        help="Compressor type to test"
    )

    parser.add_argument(
        "--test_both",
        action="store_true",
        help="Test both N-Layers and ModernBERT compressors"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of synthetic examples to create"
    )

    args = parser.parse_args()

    # Check if CUDA is available
    print("\nSystem Info:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    if args.test_both:
        # Test N-Layers
        print("\n" + "=" * 80)
        print("TESTING N-LAYERS COMPRESSOR (OSCAR Paper)")
        print("=" * 80)
        success_nlayers = run_all_tests("n_layers", args.num_examples)

        # Test ModernBERT
        print("\n" + "=" * 80)
        print("TESTING MODERNBERT COMPRESSOR (Novel Variant)")
        print("=" * 80)
        success_modernbert = run_all_tests("modernbert", args.num_examples)

        # Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - BOTH COMPRESSORS")
        print("=" * 80)
        print(f"  N-Layers:    {'✓ PASS' if success_nlayers else '✗ FAIL'}")
        print(f"  ModernBERT:  {'✓ PASS' if success_modernbert else '✗ FAIL'}")
        print("=" * 80)

        if success_nlayers and success_modernbert:
            print("\n🎉 Both compressor variants work correctly!")
        else:
            print("\n❌ One or more compressor variants failed.")

    else:
        # Test single compressor
        run_all_tests(args.compressor_type, args.num_examples)


if __name__ == "__main__":
    main()
