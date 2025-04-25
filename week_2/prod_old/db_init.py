import torch
import json
import numpy as np
import sys
from pathlib import Path
import chromadb
import random


from data_init import load_passages_from_file


def embed_texts_with_doc_tower(model, texts, batch_size=256, device="cpu", collection=None, start_id=0):
    model.to(device)
    model.eval()

    total_docs = len(texts)
    total_batches = (total_docs + batch_size - 1) // batch_size
    print(f"Embedding {total_docs} documents in {total_batches} batches (starting from ID {start_id})...")

    print_interval = total_docs // 10 if total_docs >= 10 else 1  # Print every 10% or every doc if <10

    with torch.no_grad():
        for i in range(0, total_docs, batch_size):
            batch = texts[i:i+batch_size]
            doc_embeds = model(batch, tower_type="doc")
            batch_embeddings = doc_embeds.cpu().numpy()
            batch_ids = [str(start_id + i + j) for j in range(len(batch))]

            # Add this batch directly to ChromaDB
            try:
                collection.add(documents=batch, embeddings=batch_embeddings.tolist(), ids=batch_ids)
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}")
                return False  # Return False if there was an error adding to the collection

            if (i + len(batch)) % print_interval < batch_size:
                percent = int(100 * (i + len(batch)) / total_docs)
                print(f"Progress: {percent}% ({i + len(batch)}/{total_docs} docs processed)")

    # If all batches were added successfully, return True
    print("All batches successfully added to ChromaDB collection.")
    return True



def create_or_load_chroma_db(docs_path, model, collection_name="docs"):
    print("Creating / loading ChromaDB collection...")
    # Create or load ChromaDB collection with persistence
    client = chromadb.PersistentClient(path="./chroma_db")  # This will create a 'chroma_db' directory in your current folder
    
    try:
        client.delete_collection(collection_name) #### DELETE THIS LATER
        collection = client.get_collection(collection_name)
        print(f"Loaded existing collection '{collection_name}'.")
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating with cosine distance metric...")
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    print("Collection created. Loading passages from file...")
    # Load passages from file (a list of strings)
    passages = load_passages_from_file(docs_path)
    
    # Randomly sample 50,000 passages if we have more than that - with a seed
    if len(passages) > 50000:
        print(f"Sampling 50,000 passages from total of {len(passages)} passages...")
        random.seed(42)
        passages = random.sample(passages, 50000)
    else:
        print(f"Using all {len(passages)} passages since total is less than 50,000")

    # Retrieve existing document texts from the collection (as list of strings)
    existing_texts = set(collection.get()["documents"])  # Assumes documents are just strings
    print(f"Found {len(existing_texts)} existing documents in the collection.")

    # Filter out passages with identical text that are already in the collection
    new_passages = [text for text in passages if text not in existing_texts]
    print(f"Found {len(new_passages)} new passages to add to the collection.")

    if not new_passages:
        print("No new passages to add to the collection. Skipping embedding process.")
        return True  # No new passages to add, consider it successful.
    else:
        print("Embedding new passages and storing in ChromaDB...")
        # Embed new passages
        docs_embed_success = embed_texts_with_doc_tower(model, new_passages, collection=collection)

    # view collection size
    print(f"New collection size: {collection.count()}")
    return docs_embed_success



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