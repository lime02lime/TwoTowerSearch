import torch
import json
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from week_2.data_preparation.data_prep import load_passages_from_file
from week_2.tower_model.model import DualTowerWithFC


def embed_texts_with_doc_tower(model, texts, batch_size=64, device="cpu", collection=None):
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            doc_embeds = model(batch, tower_type="doc")
            embeddings.extend(doc_embeds.cpu().numpy())

    # finally store the embeddings in the collection
    collection.add(documents=texts, embeddings=embeddings)
    return embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    data_prep_dir = Path(__file__).parent.parent / "data_preparation"
    all_docs_path = data_prep_dir / "all_docs.json"

    # Load the passages
    all_passages = load_passages_from_file(str(all_docs_path))

    # Load the trained model
    model = DualTowerWithFC()
    model.load_state_dict(torch.load("week_2/tower_model/dual_tower_model_base.pt", map_location=device))
    model.eval()

    # Initialize ChromaDB client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

    # Create or access a collection
    collection_name = 'ms_marco_documents'
    collection = client.get_or_create_collection(name=collection_name)

    # Embed and store the documents
    print("Embedding documents...")
    embed_texts_with_doc_tower(model, all_passages, device=device, collection=collection)

    # Verify the data is stored
    stored_docs = collection.get()["documents"]
    print(f"Total documents in collection: {len(stored_docs)}")


if __name__ == "__main__":
    main()
