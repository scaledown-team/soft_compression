"""
Prepare a small real dataset for ScaleDown training.

This script downloads and prepares real QA data from public datasets:
- Option 1: SQuAD (Stanford Question Answering Dataset)
- Option 2: Natural Questions (Google)
- Option 3: TriviaQA

No retrieval needed - each example comes with query + context + answer.
Perfect for quick real-world testing.
"""

import argparse
import json
from typing import List, Dict
from pathlib import Path


def prepare_squad_dataset(num_examples: int = 500, split: str = "validation") -> List[Dict]:
    """
    Prepare SQuAD dataset.

    Args:
        num_examples: Number of examples to use
        split: Dataset split ('train' or 'validation')

    Returns:
        List of ScaleDown-formatted examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return []

    print(f"\nDownloading SQuAD {split} split...")
    squad = load_dataset("squad", split=split)

    print(f"Converting {num_examples} examples to ScaleDown format...")
    data = []

    for i, example in enumerate(squad):
        if i >= num_examples:
            break

        # SQuAD format: question, context, answers
        data.append({
            "query": example["question"],
            "documents": [example["context"]],  # Single document (the context)
            "answer": example["answers"]["text"][0],  # First answer
        })

    print(f"✓ Prepared {len(data)} SQuAD examples")
    return data


def prepare_natural_questions_dataset(num_examples: int = 500) -> List[Dict]:
    """
    Prepare Natural Questions dataset.

    Args:
        num_examples: Number of examples to use

    Returns:
        List of ScaleDown-formatted examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return []

    print(f"\nDownloading Natural Questions...")
    # Use the simplified version
    nq = load_dataset("natural_questions", "default", split="train", streaming=True)

    print(f"Converting {num_examples} examples to ScaleDown format...")
    data = []

    for i, example in enumerate(nq):
        if i >= num_examples:
            break

        # Extract question and answer
        question = example["question"]["text"]

        # Get short answer if available
        if example["annotations"]["short_answers"]:
            answer_start = example["annotations"]["short_answers"][0]["start_token"]
            answer_end = example["annotations"]["short_answers"][0]["end_token"]
            tokens = example["document"]["tokens"]["token"]
            answer = " ".join(tokens[answer_start:answer_end])
        else:
            # Skip examples without short answers
            continue

        # Use document as context
        document = " ".join(example["document"]["tokens"]["token"][:512])  # First 512 tokens

        data.append({
            "query": question,
            "documents": [document],
            "answer": answer,
        })

    print(f"✓ Prepared {len(data)} Natural Questions examples")
    return data


def prepare_trivia_qa_dataset(num_examples: int = 500) -> List[Dict]:
    """
    Prepare TriviaQA dataset.

    Args:
        num_examples: Number of examples to use

    Returns:
        List of ScaleDown-formatted examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return []

    print(f"\nDownloading TriviaQA...")
    trivia = load_dataset("trivia_qa", "unfiltered", split="train")

    print(f"Converting {num_examples} examples to ScaleDown format...")
    data = []

    for i, example in enumerate(trivia):
        if i >= num_examples:
            break

        # TriviaQA has search results as "documents"
        documents = []
        for search_result in example.get("search_results", {}).get("search_context", [])[:3]:
            # Use first 3 search results
            documents.append(search_result)

        if not documents:
            # Use entity pages if no search results
            for entity_page in example.get("entity_pages", {}).get("wiki_context", [])[:3]:
                documents.append(entity_page)

        if not documents:
            continue

        data.append({
            "query": example["question"],
            "documents": documents,
            "answer": example["answer"]["value"],
        })

    print(f"✓ Prepared {len(data)} TriviaQA examples")
    return data


def prepare_hotpot_qa_dataset(num_examples: int = 500) -> List[Dict]:
    """
    Prepare HotpotQA dataset (multi-hop reasoning).

    Args:
        num_examples: Number of examples to use

    Returns:
        List of ScaleDown-formatted examples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return []

    print(f"\nDownloading HotpotQA...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="train")

    print(f"Converting {num_examples} examples to ScaleDown format...")
    data = []

    for i, example in enumerate(hotpot):
        if i >= num_examples:
            break

        # HotpotQA provides multiple context documents
        documents = []
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            # Combine sentences into document
            doc_text = " ".join(sentences)
            documents.append(f"{title}: {doc_text}")

        data.append({
            "query": example["question"],
            "documents": documents[:5],  # Use first 5 documents
            "answer": example["answer"],
        })

    print(f"✓ Prepared {len(data)} HotpotQA examples")
    return data


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare small real dataset for ScaleDown training"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["squad", "natural_questions", "trivia_qa", "hotpot_qa", "all"],
        help="Dataset to use"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=500,
        help="Number of examples to prepare"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="small_real_dataset.json",
        help="Output file path"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to use (for SQuAD)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ScaleDown: Prepare Small Real Dataset")
    print("=" * 80)

    # Prepare dataset
    if args.dataset == "squad":
        data = prepare_squad_dataset(args.num_examples, args.split)

    elif args.dataset == "natural_questions":
        data = prepare_natural_questions_dataset(args.num_examples)

    elif args.dataset == "trivia_qa":
        data = prepare_trivia_qa_dataset(args.num_examples)

    elif args.dataset == "hotpot_qa":
        data = prepare_hotpot_qa_dataset(args.num_examples)

    elif args.dataset == "all":
        # Combine multiple datasets
        print("\nPreparing mixed dataset from all sources...")
        per_dataset = args.num_examples // 4

        data = []
        data.extend(prepare_squad_dataset(per_dataset))
        data.extend(prepare_natural_questions_dataset(per_dataset))
        data.extend(prepare_trivia_qa_dataset(per_dataset))
        data.extend(prepare_hotpot_qa_dataset(per_dataset))

        print(f"\n✓ Combined {len(data)} examples from all datasets")

    else:
        print(f"Unknown dataset: {args.dataset}")
        return

    if not data:
        print("No data prepared. Exiting.")
        return

    # Split into train/eval (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # Save training data
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"\n✓ Saved {len(train_data)} training examples to {output_file}")

    # Save evaluation data
    eval_file = output_file.replace('.json', '_eval.json')
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"✓ Saved {len(eval_data)} evaluation examples to {eval_file}")

    # Print sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    sample = train_data[0]
    print(f"\nQuery: {sample['query']}")
    print(f"\nDocuments ({len(sample['documents'])}):")
    for i, doc in enumerate(sample['documents'][:2], 1):
        print(f"  {i}. {doc[:100]}...")
    print(f"\nAnswer: {sample['answer']}")
    print("=" * 80)

    print(f"\nNext steps:")
    print(f"  1. Train: python train_with_evaluation.py --train_data {output_file}")
    print(f"  2. Or: python train.py --train_data {output_file}")


if __name__ == "__main__":
    main()
