"""
Example usage of ScaleDown for training and inference.

This script demonstrates:
1. Creating synthetic training data
2. Training ScaleDown model
3. Using the model for inference
"""

import torch
from scaledown import ScaleDownConfig, ScaleDownModel
from scaledown.data.dataset import ScaleDownDataset
from scaledown.training.trainer import ScaleDownTrainer


def create_synthetic_data(num_examples=100):
    """
    Create synthetic training data for demonstration.

    In practice, you would:
    1. Retrieve documents using your retriever (e.g., SPLADE, BM25)
    2. Generate answers using a teacher LLM (e.g., Mistral-7B)
    3. Optionally, get reranking scores from a cross-encoder (e.g., DeBERTa-v3)
    """
    data = []

    for i in range(num_examples):
        example = {
            "query": f"What is example query {i}?",
            "documents": [
                f"This is document 1 for query {i}. It contains relevant information.",
                f"This is document 2 for query {i}. It has some related content.",
                f"This is document 3 for query {i}. It provides additional context.",
                f"This is document 4 for query {i}. It offers supplementary details.",
                f"This is document 5 for query {i}. It gives further explanation.",
            ],
            "answer": f"The answer to query {i} is based on the provided documents.",
            # Optional: reranking scores from teacher (e.g., DeBERTa-v3)
            "reranking_scores": [0.95, 0.7, 0.5, 0.3, 0.1],
        }
        data.append(example)

    return data


def example_training():
    """Example: Train ScaleDown model."""

    print("=" * 80)
    print("ScaleDown Training Example")
    print("=" * 80)

    # 1. Create configuration
    config = ScaleDownConfig(
        # Compressor settings
        compressor_type="n_layers",  # or "modernbert" for novel variant
        num_compressor_layers=8,      # 5, 8, or 10 for n_layers
        num_memory_tokens=8,
        compression_rate=16,

        # Generator settings
        generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,

        # Training settings
        batch_size=4,  # Small batch for demo (paper uses 128)
        num_epochs=1,
        learning_rate_generator=1e-4,
        learning_rate_compressor_nlayers=5e-5,

        # Device
        device_type="gpu",  # or "trainium" for AWS Trainium

        # Optional: Enable reranking
        enable_reranking=False,
        reranking_loss_weight=0.05,
    )

    # 2. Create synthetic data
    print("\nCreating synthetic training data...")
    train_data = create_synthetic_data(num_examples=20)
    eval_data = create_synthetic_data(num_examples=5)

    # 3. Create datasets
    print("Creating datasets...")
    train_dataset = ScaleDownDataset(train_data, config)
    eval_dataset = ScaleDownDataset(eval_data, config)

    # 4. Create model
    print("\nInitializing ScaleDown model...")
    print(f"  Compressor: {config.compressor_type}")
    print(f"  Generator: {config.generator_model_name}")
    model = ScaleDownModel(config)

    # 5. Train
    print("\nCreating trainer...")
    trainer = ScaleDownTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./example_checkpoints",
    )

    print("\nStarting training...")
    trainer.train()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("Model saved to: ./example_checkpoints/final")
    print("=" * 80)


def example_inference():
    """Example: Use trained ScaleDown model for inference."""

    print("=" * 80)
    print("ScaleDown Inference Example")
    print("=" * 80)

    # 1. Load configuration and model
    config = ScaleDownConfig(
        compressor_type="n_layers",
        num_compressor_layers=8,
        generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    )

    model = ScaleDownModel(config)

    # Load trained weights (in practice)
    # model.load_state_dict(torch.load("./example_checkpoints/final/pytorch_model.bin"))

    model.eval()

    # 2. Prepare input
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France. Located in the north-central part of the country...",
        "France is a country located in Western Europe. It has several major cities including...",
        "The Eiffel Tower is one of the most famous landmarks in Paris, attracting millions of visitors...",
    ]

    # Tokenize (simplified - use proper tokenizer in practice)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.generator_model_name)

    # This is a simplified example - see dataset.py for proper preprocessing
    query_encoding = tokenizer(query, return_tensors="pt", max_length=128, padding="max_length")

    # 3. Compress documents (simplified)
    # In practice, you'd properly format with memory tokens as shown in dataset.py

    print("\nQuery:", query)
    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc[:80]}...")

    # 4. Generate answer
    # answer = model.generate(
    #     query_input_ids=query_ids,
    #     query_attention_mask=query_mask,
    #     doc_input_ids=doc_ids,
    #     doc_attention_mask=doc_mask,
    #     memory_token_positions=mem_positions,
    #     max_new_tokens=128,
    # )

    print("\nNote: See dataset.py for proper input formatting with memory tokens")
    print("=" * 80)


def main():
    """Run examples."""

    print("\nScaleDown Examples\n")
    print("Choose an example:")
    print("1. Training example (with synthetic data)")
    print("2. Inference example")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        example_training()
    elif choice == "2":
        example_inference()
    elif choice == "3":
        example_training()
        print("\n" + "=" * 80 + "\n")
        example_inference()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
