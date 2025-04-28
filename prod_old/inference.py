import torch
import chromadb
from db_init import search_collection


def run_application(model, collection_name):
    """
    Simple loop that queries the user and returns the top k closest documents from the ChromaDB.
    """
    # Create ChromaDB client and load the collection
    client = chromadb.PersistentClient(path="./chroma_db")  # Use the same persistent client
    collection = client.get_or_create_collection(collection_name)  # This loads or creates the collection based on name

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
                
                for idx, (doc, dist) in enumerate(zip(closest_docs[0], distances[0])):  # Access the first element of the outer lists
                    print(f"\n{idx + 1}.")
                    print(f"   Document Text: {doc[:1000]}...")  # No need for [0] since we're already accessing the inner list
                    print(f"   Distance: {dist:.4f}")  # No need for [0] since we're already accessing the inner list
                    print(f"{'. '*40}")
                    

            else:
                print("No results found.")

        except Exception as e:
            print(f"Error during query processing: {e}")
