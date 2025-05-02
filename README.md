# Two Tower Search

A PyTorch implementation of a dual-tower neural network for semantic search using sentence transformers. This model architecture is designed to learn the semantic similarity between queries and documents.

## Overview

The project implements a dual-tower architecture where:
- One tower processes queries
- Another tower processes documents
- Both towers share the same base sentence transformer model but have separate fully connected layers
- The model learns to create embeddings that bring relevant queries and documents closer in the embedding space

## Model Architecture

### DualTowerWithFC
- Base model: `multi-qa-MiniLM-L6-cos-v1` (384-dimensional embeddings)
- Each tower consists of:
  - A shared sentence transformer for initial embeddings
  - Three fully connected layers with ReLU activations and dropout
  - Layer normalization for consistent embedding magnitudes

### Loss Function
- Uses margin-based triplet loss to learn semantic similarity
- Evaluated using the query, a positive (correct) document and a negative (incorrect) document.
- This approachg helps to increase the differentiability between similar documents.

## Requirements

- PyTorch
- sentence-transformers
- wandb (for experiment tracking)
- FastAPI
- ChromaDB
- uvicorn

## Application

The model is deployed as a FastAPI application that provides a REST API for document search. The application uses ChromaDB for efficient vector storage and retrieval.

### API Endpoints

- `GET /`: Welcome message
- `POST /search`: Search for similar documents
  - Input: Query text and number of results (k)
  - Output: List of similar documents with their distances

### Running the Application

```bash
uvicorn prod.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with automatic API documentation at `/docs`.

## Features

- Shared sentence transformer backbone for efficient learning
- Separate fully connected layers for query and document processing
- Dropout layers for regularization
- Normalized embeddings for consistent similarity calculations
- Wandb integration for experiment tracking
- FastAPI deployment with ChromaDB integration
- CORS support for cross-origin requests

## Future Work:

- Could try to embed the doc with a more high-dimensional base embedding model, as the documents can be long and complex. As long as the subsequent MLP reduces the dimensionality to match with the query tower, this should be compatible and might allow us to capture more relevant information from the document.
