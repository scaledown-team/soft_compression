"""
Teacher LLM utilities for ScaleDown dataset generation.

This module implements the teacher distillation pipeline from the OSCAR paper:
- Generate answers using teacher LLM (Mistral-7B)
- Sentence-level distillation (no ground truth labels needed)
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class TeacherLLM:
    """
    Teacher LLM for generating training answers.

    Following OSCAR paper Section 4.2:
    - Uses Mistral-7B-Instruct as teacher
    - Generates answers given query + retrieved documents
    - Student model learns to mimic teacher's outputs
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
    ):
        """
        Initialize teacher LLM.

        Args:
            model_name: HuggingFace model name for teacher
            device: Device to run generation on
            load_in_8bit: Use 8-bit quantization to save memory
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.model.eval()

    def format_prompt(
        self,
        query: str,
        documents: List[Dict[str, str]],
        max_doc_length: int = 200,
    ) -> str:
        """
        Format query and documents into a prompt for the teacher.

        Args:
            query: User query
            documents: Retrieved documents with 'title', 'text' fields
            max_doc_length: Maximum length per document in tokens (approx)

        Returns:
            Formatted prompt string
        """
        # Format documents
        doc_strings = []
        for i, doc in enumerate(documents, 1):
            # Truncate document text
            text = doc['text']
            words = text.split()[:max_doc_length]
            truncated_text = ' '.join(words)

            doc_strings.append(
                f"Document {i}: {doc['title']}\n{truncated_text}"
            )

        docs_text = "\n\n".join(doc_strings)

        # Create prompt following Mistral-Instruct format
        prompt = f"""[INST] You are a helpful assistant. Answer the following question based on the provided documents.

Question: {query}

Documents:
{docs_text}

Provide a concise and accurate answer based on the documents above. [/INST]"""

        return prompt

    def generate_answer(
        self,
        query: str,
        documents: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate answer for a single query-document set.

        Args:
            query: User query
            documents: Retrieved documents
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated answer string
        """
        # Format prompt
        prompt = self.format_prompt(query, documents)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode (remove prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text.strip()

    def generate_answers_batch(
        self,
        queries_with_docs: List[Dict[str, any]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, any]]:
        """
        Generate answers for a batch of query-document sets.

        Args:
            queries_with_docs: List of dicts with 'query' and 'documents' fields
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Input list with added 'answer' field for each item
        """
        results = []

        for i, item in enumerate(queries_with_docs):
            print(f"Generating answer {i+1}/{len(queries_with_docs)}...")

            answer = self.generate_answer(
                query=item['query'],
                documents=item['documents'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Add answer to item
            result = item.copy()
            result['answer'] = answer
            results.append(result)

        return results


def generate_teacher_answers(
    queries_with_docs: List[Dict[str, any]],
    teacher_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 128,
    load_in_8bit: bool = False,
) -> List[Dict[str, any]]:
    """
    Main function to generate teacher answers for training data.

    This follows the OSCAR paper's distillation approach:
    1. Format query + retrieved documents into prompt
    2. Generate answer using teacher LLM (Mistral-7B)
    3. Student learns to mimic teacher's outputs with compressed documents

    Args:
        queries_with_docs: List of dicts with 'query' and 'documents' fields
        teacher_model_name: HuggingFace model name for teacher
        max_new_tokens: Maximum tokens to generate
        load_in_8bit: Use 8-bit quantization

    Returns:
        Input list with added 'answer' field for each item
    """
    # Initialize teacher
    print(f"Loading teacher model: {teacher_model_name}...")
    teacher = TeacherLLM(
        model_name=teacher_model_name,
        load_in_8bit=load_in_8bit,
    )

    # Generate answers
    print(f"Generating answers for {len(queries_with_docs)} queries...")
    results = teacher.generate_answers_batch(
        queries_with_docs,
        max_new_tokens=max_new_tokens,
    )

    print("Answer generation complete!")
    return results
