import pandas as pd
import numpy as np
import os

from datasets import load_dataset
import json
import random
from typing import List, Tuple



def create_triplets(dataset, include_hard_negatives=False, num_hard_negatives=3):
    """
    Create (query, positive_passage, negative_passage) triplets from the given dataset.
    For training, add additional triplets with in-row "hard" negatives.
    
    Args:
        dataset (list of dict): Each item should have 'Query' and 'Passages' keys. 
                                'Passages' must contain 'is_selected' and 'passage_text'.
        include_hard_negatives (bool): Whether to add hard negative samples (for training only)
        num_hard_negatives (int): Number of hard negatives to sample per positive (if available)
    
    Returns:
        list of tuples: Each tuple is (query, positive_passage, negative_passage)
    """
    all_passages = []

    # Pre-collect all passages for negative sampling
    for row in dataset:
        all_passages.extend(row['passages']['passage_text'])

    triplets = []

    for row in dataset:
        query = row['query']
        passages = row['passages']['passage_text']
        labels = row['passages']['is_selected']

        # Find the index of the positive passage
        if 1 not in labels:
            continue  # Skip if no positive passage
        pos_index = labels.index(1)
        positive = passages[pos_index]

        # Select a random negative passage (ensuring it's not from the same row)
        while True:
            negative = random.choice(all_passages)
            if negative != positive and negative not in passages:
                break

        triplets.append((query, positive, negative))

        if include_hard_negatives:
            # Create additional triplets with in-row negatives (if available)
            non_selected_indices = [i for i, label in enumerate(labels) if label == 0]
            
            if non_selected_indices:
                # Sample up to num_hard_negatives unique in-row negatives
                # (or fewer if not enough are available)
                sampled_indices = random.sample(
                    non_selected_indices, 
                    min(num_hard_negatives, len(non_selected_indices))
                )
                
                for neg_index in sampled_indices:
                    in_row_negative = passages[neg_index]
                    triplets.append((query, positive, in_row_negative))

    return triplets



def save_triplets_to_json(triplets: List[Tuple[str, str, str]], output_file: str) -> None:
    """
    Save triplets to a JSON file.
    
    Args:
        triplets: List of (query, positive_passage, negative_passage) tuples
        output_file: Path to save the JSON file
    """
    # Convert tuples to dictionaries for better readability
    triplets_dict = [
        {
            "query": query,
            "positive_passage": pos,
            "negative_passage": neg
        }
        for query, pos, neg in triplets
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triplets_dict, f, ensure_ascii=False, indent=2)


def load_triplets_from_json(input_file: str) -> List[Tuple[str, str, str]]:
    """
    Load triplets from a JSON file.
    
    Args:
        input_file: Path to the JSON file
        
    Returns:
        List of (query, positive_passage, negative_passage) tuples
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        triplets_dict = json.load(f)
    
    # Convert dictionaries back to tuples
    triplets = [
        (item["query"], item["positive_passage"], item["negative_passage"])
        for item in triplets_dict
    ]
    
    return triplets


def save_passages_to_file(passages, file_path):
    """Save the passages to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(passages, file)
    print(f"Passages saved to {file_path}")


def load_passages_from_file(file_path):
    """Load passages from a JSON file."""
    with open(file_path, 'r') as file:
        passages = json.load(file)
    print(f"Loaded {len(passages)} passages from {file_path}")
    return passages


def generate_documents_if_needed(docs_path):
    if not os.path.exists(docs_path):
        print(f"Documents file not found at {docs_path}. Generating...")
        # Implement document generation logic here
        train_dataset = load_dataset('ms_marco', 'v2.1', split='train')
        test_dataset = load_dataset('ms_marco', 'v2.1', split='validation')

        # create triplets from dataset
        train_triplets = create_triplets(train_dataset, include_hard_negatives=False)
        test_triplets = create_triplets(test_dataset, include_hard_negatives=False)
        print(f"Generated {len(train_triplets)} train triplets and {len(test_triplets)} test triplets. Now saving to files...")

        # save triplets to file
        save_triplets_to_json(train_triplets, 'train_triplets.json')
        save_triplets_to_json(test_triplets, 'test_triplets.json')
        print(f"Saved triplets to files. Now saving passages to file...")

        # save passages to file
        #save_passages_to_file(train_dataset['passages']['passage_text'], 'train_passages.json')
        test_passages = [passage for row in test_dataset for passage in row['passages']['passage_text']]
        save_passages_to_file(test_passages, 'test_passages.json')
        print(f"Saved test passages to file.")

    else:
        print(f"Documents file found at {docs_path}")
