import os
from model_init import load_model
import torch
from data_init import generate_documents_if_needed
from db_init import create_or_load_chroma_db
from inference import run_application   # You define how this is served (e.g. FastAPI)


def main():

    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Step 1: Load model
        model_weights_path = "dual_tower_model_base_384D.pt"
        model = load_model(model_weights_path=model_weights_path, device=device)

        # Step 2: Prepare documents (MS MARCO) if not done already
        docs_path = "test_passages.json"
        generate_documents_if_needed(docs_path)

        # Step 3: Create or load ChromaDB vector DB
        collection_name="docs"
        docs_embed_success = create_or_load_chroma_db(docs_path, model, collection_name)
        if not docs_embed_success: raise Exception("Failed to create or load ChromaDB vector DB with embeddings")

    except Exception as e:
        print(f"Error during startup: {e}")

    try:
        run_application(model, collection_name)
    except KeyboardInterrupt:
        print("\nApplication exited by user (Ctrl+C).")
    except Exception as e:
        print(f"Error during application execution: {e}")


if __name__ == "__main__":
    main()
