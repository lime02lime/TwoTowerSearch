import chromadb
import torch
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)
from week_2.tower_model.model import DualTowerWithFC


# Load the model and ChromaDB client
def load_model_and_client(device="cpu"):
    # Use your custom model
    model = DualTowerWithFC()
    model.load_state_dict(torch.load("week_2/tower_model/dual_tower_model_base_384D.pt", map_location=device))
    model.to(device)
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        # Access the collection
        collection = client.get_collection('ms_marco_documents')
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return None, None
    
    return model, collection


# Perform the search and return the closest document
def search_collection(query, model, collection, device="cpu"):
    model.eval()
    with torch.no_grad():
        query_embedding = model([query], tower_type="query").to(device).cpu().numpy()
    
    # Search for the most similar document
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1  # We only want the closest match
    )
    
    if results['documents']:
        return results['documents'][0], results['distances'][0]
    else:
        return None, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and ChromaDB client
    model, collection = load_model_and_client(device=device)
    if collection is None:
        print("Failed to load ChromaDB client. Exiting.")
        return
    

    print("-" * 40)
    print("Welcome to the document search. Type 'exit' or ctrl+C to quit the program.")
    print("-" * 40)

    
    try:
        while True:
            # Prompt the user for a search query
            query = input("\nEnter a search query: ")
            
            if query.lower() == "exit":
                print("Exiting the search.")
                break
            
            # Search for the closest matching document
            closest_doc, distance = search_collection(query, model, collection, device)
            
            if closest_doc:
                print("\nClosest match found:")
                print(f"Document: {closest_doc}")
                print(f"Distance: {distance}")
                print("-" * 40)
            else:
                print("No matching document found.")
    
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\nSearch interrupted. Exiting the program.")

if __name__ == "__main__":
    main()