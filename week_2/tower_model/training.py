import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from week_2.data_preparation.data_prep import load_triplets_from_json
from model import DualTowerWithFC, TripletLoss

class TripletDataset(Dataset):
    def __init__(self, triplets):
        """
        Args:
            triplets: List of (query, positive_doc, negative_doc) tuples
        """
        self.queries = [t[0] for t in triplets]
        self.pos_docs = [t[1] for t in triplets]
        self.neg_docs = [t[2] for t in triplets]
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx], self.pos_docs[idx], self.neg_docs[idx]

def train(model, triplets, criterion, optimizer, num_epochs=5, batch_size=32):
    """
    Train the dual tower model using triplet loss
    """
    model.train()
    
    # Create dataset and dataloader
    dataset = TripletDataset(triplets)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: ([q for q, _, _ in x], 
                             [p for _, p, _ in x],
                             [n for _, _, n in x])
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (batch_queries, batch_pos_docs, batch_neg_docs) in enumerate(dataloader):
            # Get embeddings for the entire batch
            query_emb = model(batch_queries, tower_type="query")
            pos_doc_emb = model(batch_pos_docs, tower_type="doc")
            neg_doc_emb = model(batch_neg_docs, tower_type="doc")
            
            # Calculate loss
            loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Changed from 100 to 10 since we'll have fewer batches
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Batch sizes - Query: {len(batch_queries)}, Pos: {len(batch_pos_docs)}, Neg: {len(batch_neg_docs)}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} complete. Average Loss: {avg_loss:.4f}')

def main():
    # Get the path to triplets.json
    data_prep_dir = Path(__file__).parent.parent / "data_preparation"
    triplets_path = data_prep_dir / "triplets.json"
    
    print("Loading triplets...")
    print(f"Looking for triplets at: {triplets_path}")
    triplets = load_triplets_from_json(str(triplets_path))
    print(f"Loaded {len(triplets)} triplets")

    # print triplets shape
    print(f"Triplets shape: {len(triplets)}")
    exit()
    
    print("Initializing model...")
    model = DualTowerWithFC()
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    

    print("Starting training...")
    train(model, triplets, criterion, optimizer)
    
    print("Training complete!")
    
    # Save the model in the same directory as the script
    model_save_path = Path(__file__).parent / "dual_tower_model.pt"
    torch.save(model.state_dict(), str(model_save_path))
    print(f"Model saved to {model_save_path}!")

if __name__ == "__main__":
    main() 