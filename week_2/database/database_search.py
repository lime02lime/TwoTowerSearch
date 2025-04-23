import chromadb
import torch
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Load the model and ChromaDB client
def load_model_and_client(device="cpu"):
    # Initialize model and load the SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model.to(device)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Access the collection
    collection_name = 'ms_marco_documents'
    collection = client.get_collection(collection_name)
    
    return model, collection


# Perform the search and return the closest document
def search_collection(query, model, collection, device="cpu"):
    # Generate embedding for the query
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    
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