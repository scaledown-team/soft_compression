"""
Example: Generate training dataset for ScaleDown.

This script demonstrates how to use the dataset generation utilities
to create training data following the OSCAR paper's methodology.
"""

import json
from scaledown.data.retrieval import retrieve_documents_for_queries
from scaledown.data.teacher import generate_teacher_answers
from scaledown.data.reranker import add_reranking_scores


def example_synthetic_dataset():
    """
    Example 1: Create a small synthetic dataset for testing.

    This doesn't require Wikipedia-KILT or any external data.
    """
    print("=" * 80)
    print("Example 1: Synthetic Dataset")
    print("=" * 80)

    # Create synthetic data directly
    data = [
        {
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.",
                "Machine learning algorithms use statistical techniques to identify patterns in data.",
            ],
            "answer": "Machine learning is a subset of AI that enables computers to learn from data using statistical techniques.",
        },
        {
            "query": "How does photosynthesis work?",
            "documents": [
                "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                "Plants use chlorophyll to capture sunlight and convert carbon dioxide and water into glucose.",
                "The process occurs in chloroplasts and produces oxygen as a byproduct.",
            ],
            "answer": "Photosynthesis is how plants convert light energy into chemical energy using chlorophyll, producing glucose and oxygen.",
        },
    ]

    # Save
    output_file = "synthetic_train_data.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Created {len(data)} synthetic examples")
    print(f"✓ Saved to: {output_file}")
    print("\nUse for training:")
    print(f"  python train.py --train_data {output_file}")


def example_with_teacher_generation():
    """
    Example 2: Generate dataset with teacher LLM.

    This uses a teacher LLM to generate answers for pre-retrieved documents.
    """
    print("\n" + "=" * 80)
    print("Example 2: Teacher LLM Generation")
    print("=" * 80)

    # Pre-retrieved query-document pairs
    # In practice, these would come from your retriever (SPLADE, BM25, etc.)
    queries_with_docs = [
        {
            "query": "What is deep learning?",
            "documents": [
                {
                    "title": "Deep Learning",
                    "text": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers.",
                },
                {
                    "title": "Neural Networks",
                    "text": "Neural networks are computing systems inspired by biological neural networks in the brain.",
                },
                {
                    "title": "AI History",
                    "text": "Deep learning has revolutionized AI since the 2010s, enabling breakthroughs in computer vision and NLP.",
                },
            ],
        },
    ]

    print("\nGenerating answers with teacher LLM (Mistral-7B)...")
    print("Note: This requires GPU and will download ~15GB model")
    print("Skipping actual generation in this example.\n")

    # Uncomment to actually run:
    # training_data = generate_teacher_answers(
    #     queries_with_docs,
    #     teacher_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     load_in_8bit=True,  # Use 8-bit for lower memory
    # )
    #
    # # Save
    # output_file = "teacher_generated_data.json"
    # with open(output_file, "w") as f:
    #     json.dump(training_data, f, indent=2)
    #
    # print(f"✓ Generated {len(training_data)} examples with teacher")
    # print(f"✓ Saved to: {output_file}")


def example_with_reranking():
    """
    Example 3: Add reranking scores for joint compression + reranking.

    This adds DeBERTa-v3 reranking scores to enable joint training.
    """
    print("=" * 80)
    print("Example 3: Adding Reranking Scores")
    print("=" * 80)

    # Query-document pairs with answers
    queries_with_docs = [
        {
            "query": "What is quantum computing?",
            "documents": [
                {
                    "text": "Quantum computing uses quantum mechanics to perform computations.",
                },
                {
                    "text": "Unlike classical computers, quantum computers use qubits.",
                },
                {
                    "text": "Quantum computers can solve certain problems exponentially faster.",
                },
            ],
            "answer": "Quantum computing uses quantum mechanics and qubits to perform computations faster than classical computers.",
        },
    ]

    print("\nAdding reranking scores with DeBERTa-v3...")
    print("Note: This requires GPU and will download the reranker model")
    print("Skipping actual reranking in this example.\n")

    # Uncomment to actually run:
    # queries_with_reranking = add_reranking_scores(
    #     queries_with_docs,
    #     reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    # )
    #
    # # Format for training
    # training_data = []
    # for item in queries_with_reranking:
    #     training_data.append({
    #         "query": item["query"],
    #         "documents": [doc["text"] for doc in item["documents"]],
    #         "answer": item["answer"],
    #         "reranking_scores": item["reranking_scores"],
    #     })
    #
    # # Save
    # output_file = "reranking_train_data.json"
    # with open(output_file, "w") as f:
    #     json.dump(training_data, f, indent=2)
    #
    # print(f"✓ Added reranking scores to {len(training_data)} examples")
    # print(f"✓ Saved to: {output_file}")


def example_full_pipeline():
    """
    Example 4: Full pipeline with retrieval, reranking, and teacher generation.

    This is closest to the OSCAR paper's approach.
    """
    print("=" * 80)
    print("Example 4: Full Pipeline (OSCAR Paper)")
    print("=" * 80)

    print("\nFull pipeline requires:")
    print("  1. Wikipedia-KILT corpus (~35GB)")
    print("  2. SPLADE-v3 retriever")
    print("  3. DeBERTa-v3 reranker")
    print("  4. Mistral-7B teacher LLM")
    print("\nUse the command-line tool instead:")
    print("\n  python -m scaledown.data.prepare_dataset \\")
    print("    --download_ms_marco \\")
    print("    --corpus_path kilt_knowledgesource.json \\")
    print("    --output_file train_data.json \\")
    print("    --enable_reranking")
    print("\nSee DATASET_PREPARATION.md for detailed instructions.")


def main():
    """Run all examples."""
    print("\nScaleDown Dataset Generation Examples\n")

    # Example 1: Synthetic (always safe to run)
    example_synthetic_dataset()

    # Example 2: Teacher generation (commented out - requires GPU)
    example_with_teacher_generation()

    # Example 3: Reranking (commented out - requires GPU)
    example_with_reranking()

    # Example 4: Full pipeline (requires external data)
    example_full_pipeline()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("\nNext steps:")
    print("  1. Review DATASET_PREPARATION.md for detailed instructions")
    print("  2. Generate your training data")
    print("  3. Train ScaleDown model with: python train.py --train_data <your_data.json>")
    print("=" * 80)


if __name__ == "__main__":
    main()
