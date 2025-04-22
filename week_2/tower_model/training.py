import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

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


def save_model_checkpoint(model, epoch):
    model_path = f"dual_tower_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact('dual_tower_model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)



def train(model, triplets, criterion, optimizer, device, num_epochs=5, batch_size=256, test_triplets=None):
    """
    Train the dual tower model using triplet loss and evaluate on the test set if provided.
    """

    wandb.init(entity="emilengdahl", project="TwoTowerSearch", config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "model": "DualTowerWithFC",
        "margin": criterion.margin,
        "optimizer": "Adam",
        "lr": optimizer.param_groups[0]['lr']
    })

    model.train()
    
    # Create dataset and dataloader for training
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
            # Get embeddings
            query_emb = model(batch_queries, tower_type="query")
            pos_doc_emb = model(batch_pos_docs, tower_type="doc")
            neg_doc_emb = model(batch_neg_docs, tower_type="doc")
            
            # Compute loss
            loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})
        print(f'Epoch {epoch} complete. Average Loss: {avg_loss:.4f}')

        # Save model after each epoch
        save_model_checkpoint(model, epoch)

        # Optional: Evaluate on test set
        if test_triplets is not None:
            test_loss = evaluate(model, test_triplets, criterion, device, batch_size)
            wandb.log({"epoch": epoch, "test_loss": test_loss})


@torch.no_grad()
def evaluate(model, triplets, criterion, device, batch_size=256):
    model.eval()
    
    dataset = TripletDataset(triplets)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: ([q for q, _, _ in x], 
                              [p for _, p, _ in x],
                              [n for _, _, n in x])
    )

    total_loss = 0
    for batch_queries, batch_pos_docs, batch_neg_docs in dataloader:
        query_emb = model(batch_queries, tower_type="query")
        pos_doc_emb = model(batch_pos_docs, tower_type="doc")
        neg_doc_emb = model(batch_neg_docs, tower_type="doc")
        
        loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation on test set — Avg Loss: {avg_loss:.4f}")
    wandb.log({"test_loss": avg_loss})
    return avg_loss



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the path to triplets.json
    data_prep_dir = Path(__file__).parent.parent / "data_preparation"
    train_triplets_path = data_prep_dir / "train_triplets.json"
    test_triplets_path = data_prep_dir / "test_triplets.json"
    
    print("Loading triplets...")
    print(f"Looking for triplets at: {train_triplets_path} and {test_triplets_path}")
    train_triplets = load_triplets_from_json(str(train_triplets_path))
    test_triplets = load_triplets_from_json(str(test_triplets_path))
    print(f"Loaded {len(train_triplets)} train triplets")
    print(f"Loaded {len(test_triplets)} test triplets")

    
    print("Initializing model...")
    model = DualTowerWithFC().to(device)
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("Starting training...")
    train(model, train_triplets, criterion, optimizer, device=device, test_triplets=test_triplets)
    
    print("Training complete!")
    
    # Save the model in the same directory as the script
    model_save_path = Path(__file__).parent / "dual_tower_model.pt"
    torch.save(model.state_dict(), str(model_save_path))
    print(f"Model saved to {model_save_path}!")

if __name__ == "__main__":
    main()