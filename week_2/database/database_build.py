import torch
import json
import numpy as np
import sys
from pathlib import Path
import chromadb

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from week_2.data_preparation.data_prep import load_passages_from_file
from week_2.tower_model.model import DualTowerWithFC


def embed_texts_with_doc_tower(model, texts, batch_size=256, device="cpu", collection=None, start_id=0):
    model.to(device)
    model.eval()
    embeddings = []
    ids = []

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
            collection.add(documents=batch, embeddings=batch_embeddings.tolist(), ids=batch_ids)

            if (i + len(batch)) % print_interval < batch_size:
                percent = int(100 * (i + len(batch)) / total_docs)
                print(f"Progress: {percent}% ({i + len(batch)}/{total_docs} docs processed)")


    print("Storing embeddings in ChromaDB collection...")
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(f"Added {len(ids)} documents to the collection.")

    return embeddings




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    data_prep_dir = Path(__file__).parent.parent / "data_preparation"
    all_docs_path = data_prep_dir / "msmarco_v1_docs.json"

    # Load the passages
    all_passages = load_passages_from_file(str(all_docs_path))[:500_000]  # Load only the first 5 million for testing

    # Load the trained model
    model = DualTowerWithFC()
    model.load_state_dict(torch.load("week_2/tower_model/dual_tower_model_base_384D.pt", map_location=device))
    model.eval()

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or access a collection
    collection_name = 'ms_marco_documents'
    collection = client.get_or_create_collection(name=collection_name)

    # Determine how many documents are already in the collection
    existing_docs_count = collection.count()
    print(f"Collection already contains {existing_docs_count} documents")

    # Embed only new documents or all (depending on your use case)
    embed_texts_with_doc_tower(model, all_passages, batch_size=256, device=device, collection=collection, start_id=existing_docs_count)

    # Verify the data is stored
    stored_docs = collection.get()["documents"]
    print(f"Total documents in collection: {len(stored_docs)}")


if __name__ == "__main__":
    main()
