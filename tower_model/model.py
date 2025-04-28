import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import wandb


class DualTowerWithFC(nn.Module):
    def __init__(self, model_name='multi-qa-MiniLM-L6-cos-v1', embedding_dim=384, hidden_dim=384):
        super(DualTowerWithFC, self).__init__()
        
        self.embedding_model = SentenceTransformer(model_name) # 384 neurons / nodes

        self.fc_query = nn.Sequential( # Fully connected layers for the query tower
            nn.Linear(embedding_dim, hidden_dim), # 384 neurons / nodes -> 384 neurons / nodes
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), # 384 neurons / nodes -> 384 neurons / nodes
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim), # 384 neurons / nodes -> 384 neurons / nodes
            nn.ReLU(),
        )
        
        self.fc_doc = nn.Sequential(  # Fully connected layers for the document tower
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, input_text, tower_type="query"):
        # Obtain embeddings from the pre-trained model
        embeddings = self.embedding_model.encode(input_text, convert_to_tensor=True)
        
        # Pass through the corresponding fully connected layers
        if tower_type == "query":
            embeddings = self.fc_query(embeddings)
        elif tower_type == "doc":
            embeddings = self.fc_doc(embeddings)
        
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin)
    
    def forward(self, query_embedding, positive_embedding, negative_embedding):
        return self.loss_fn(query_embedding, positive_embedding, negative_embedding)



