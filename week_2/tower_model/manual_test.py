import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from model import DualTowerWithFC  # Import your model

# Add the project root to Python path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from week_2.data_preparation.data_prep import load_triplets_from_json

def get_query_embeddings(model, queries, device):
    # Get embeddings for queries
    model.eval()  # Make sure the model is in evaluation mode
    with torch.no_grad():
        query_emb = model(queries, tower_type="query")
    return query_emb

def get_doc_embeddings(model, docs, device):
    # Get embeddings for documents
    model.eval()  # Make sure the model is in evaluation mode
    with torch.no_grad():
        doc_emb = model(docs, tower_type="doc")
    return doc_emb

def compute_cosine_similarity(query_emb, doc_emb):
    # Cosine similarity between query and document embeddings
    cosine_sim = cosine_similarity(query_emb, doc_emb)
    return cosine_sim

def get_most_similar_docs(query_emb, doc_emb):
    # Get the index of the most similar document for each query
    cosine_sim = compute_cosine_similarity(query_emb, doc_emb)
    most_similar_idx = np.argmax(cosine_sim, axis=1)  # index of most similar doc for each query
    return most_similar_idx, cosine_sim

def main():
    # Set the device for running on GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the path to triplets.json
    data_prep_dir = Path(__file__).parent.parent / "data_preparation"
    test_triplets_path = data_prep_dir / "test_triplets.json"
    
    print("Loading test triplets...")
    test_triplets = load_triplets_from_json(str(test_triplets_path))
    print(f"Loaded {len(test_triplets)} test triplets")
    
    # Extract queries and documents
    queries = [t[0] for t in test_triplets]
    docs = [t[1] for t in test_triplets] + [t[2] for t in test_triplets]  # All docs (positive + negative)
    
    print("Initializing model...")
    model = DualTowerWithFC().to(device)
    model.load_state_dict(torch.load("week_2/tower_model/dual_tower_model.pt"))  # Load the trained model
    model.eval()
    
    print("Extracting embeddings for queries and docs...")
    # Get embeddings for queries and documents
    query_embeddings = get_query_embeddings(model, queries, device)
    doc_embeddings = get_doc_embeddings(model, docs, device)

    # Convert embeddings to CPU and numpy arrays for cosine similarity calculation
    query_embeddings_np = query_embeddings.cpu().numpy()
    doc_embeddings_np = doc_embeddings.cpu().numpy()

    print("Computing similarities...")
    most_similar_idx, cosine_sim = get_most_similar_docs(query_embeddings_np, doc_embeddings_np)
    
    # Display the results for the first few queries
    for i in range(5):  # Displaying top 5 queries and their most similar document
        print(f"Query {i}: {queries[i]}")
        print(f"Cosine Similarity with Documents: {cosine_sim[i]}")
        most_similar_doc_idx = most_similar_idx[i]
        print(f"Most Similar Doc Index: {most_similar_doc_idx}")
        print(f"Document: {docs[most_similar_doc_idx]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
