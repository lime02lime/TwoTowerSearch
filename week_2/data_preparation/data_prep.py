import json
from typing import List, Tuple

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