from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import chromadb
from db_init import search_collection
import os

app = FastAPI(
    title="Document Search API",
    description="API for searching documents using a dual-tower model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    text: str
    distance: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

# Global variables to store model and collection
model = None
collection = None

@app.on_event("startup")
async def startup_event():
    global model, collection
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_weights_path = "dual_tower_model_base_384D.pt"
        from model_init import load_model
        model = load_model(model_weights_path=model_weights_path, device=device)
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("docs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize application: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Document Search API"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not model or not collection:
        raise HTTPException(status_code=500, detail="Application not properly initialized")
    
    try:
        closest_docs, distances = search_collection(
            request.query, 
            model, 
            collection, 
            k=request.k
        )
        
        if not closest_docs:
            return SearchResponse(results=[])
            
        results = [
            SearchResult(text=doc, distance=float(dist))
            for doc, dist in zip(closest_docs[0], distances[0])
        ]
        
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 