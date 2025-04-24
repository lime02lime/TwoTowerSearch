import torch


# Perform the search and return the closest document
def search_collection(query, model, collection, device="cpu", k=5):
    model.eval()
    with torch.no_grad():
        query_embedding = model([query], tower_type="query").to(device).cpu().numpy()
    
    # Search for the k most similar documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k  # Retrieve top k closest matches
    )
    
    if results['documents']:
        return results['documents'], results['distances']
    else:
        return None, None


def run_application(model, collection):
    """
    Simple loop that queries the user and returns the top k closest documents from the ChromaDB.
    """

    print(f"\n{'='*40}\nAll setup complete! Application ready.")

    # prompt user for k
    k = input("Enter the number of results to return: ").strip()

    # Ensure 'k' is a valid integer
    # Ensure 'k' is a valid positive integer
    while not k.isdigit() or int(k) <= 0:
        print("Please enter a valid positive integer for k.")
        k = input("Enter the number of closest documents to retrieve (e.g., 5): ").strip()

    k = int(k)  # Convert to integer

    while True:
        # Prompt the user for a query
        print(f"\n{'='*40}:")
        query = input("Enter your query (or 'exit' to quit): ").strip()
        
        if query.lower() == "exit":
            print("Exiting the application.")
            break

        try:
            # Perform the search
            closest_docs, distances = search_collection(query, model, collection, k=k)
            
            # Print the top k results with better formatting
            if closest_docs:
                print(f"\n{'-'*40}")
                print(f"Top {k} Results for Query: '{query}'")
                
                for idx, (doc, dist) in enumerate(zip(closest_docs, distances)):
                    print(f"\n{idx + 1}.")
                    print(f"   Document Text: {doc['text'][:500]}...")  # Print only the first 500 characters for readability
                    print(f"   Distance: {dist:.4f}")
                    print(f"{'. '*40}")
                    

            else:
                print("No results found.")

        except Exception as e:
            print(f"Error during query processing: {e}")
