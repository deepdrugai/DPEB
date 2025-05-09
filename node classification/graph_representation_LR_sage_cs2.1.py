import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"
import sys
# sys.stdout = open("console_outputs/console_output_sage_cs_lr-3.txt", "w")
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
from dgl.nn import GraphConv
from torch.nn.functional import normalize

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import numpy as np
import torch
import dgl

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# #--------------------alphafold------------------#
# # Load Graph Data
# with open("/data/saiful/ePPI/alphafold_ePPI_graph_with_map.pkl", "rb") as f:
#     data = pickle.load(f)

# g = data["graph"].to(device)
# protein_to_idx = data["protein_to_idx"]
# g = dgl.add_self_loop(g)
# g.ndata["alphafold_feat"] = F.normalize(g.ndata["alphafold_feat"], p=2, dim=1).to(device)

#------------------# load all four  embeddings  ---------------#
import pickle
# load all four  embeddings 
with open("/data/saiful/ePPI/all_ePPI_embeddings_graph_with_map.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the graph and mapping
g = data["graph"]
g = data["graph"].to(device)

protein_to_idx = data["protein_to_idx"]

print("Graph and protein mapping loaded successfully!")
print("Number of nodes in graph:", g.num_nodes())
print("Example mapping:", list(protein_to_idx.items())[:5])  # Print first 5 mappings

print("g", g)
g = dgl.add_self_loop(g)
# g = get_subgraph(g)

# # # Normalize and concatenate features
g.ndata["protvec"] = normalize(g.ndata["protvec"], p=2, dim=1)
g.ndata["biovec"] = normalize(g.ndata["biovec"], p=2, dim=1)
g.ndata["alphafold"] = normalize(g.ndata["alphafold"], p=2, dim=1)
g.ndata["esm"] = normalize(g.ndata["esm"], p=2, dim=1)

import torch

# Concatenate all embeddings along the feature dimension (dim=1)
g.ndata['concatenated'] = torch.cat([
    g.ndata['protvec'],   # shape: [21435, 300]
    g.ndata['biovec'],    # shape: [21435, 1024]
    g.ndata['alphafold'], # shape: [21435, 384]
    g.ndata['esm']        # shape: [21435, 1280]
], dim=1)

# Verify the new feature was added
print(g.ndata['concatenated'].shape)  # Should output: torch.Size([21435, 2988])

#------------------# load all four  embeddings ---------------#


# Load Labels
df = pd.read_csv("/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_enzym_info_all.csv")
df["Enzyme"] = df["Enzyme"].astype(int)

node_labels = torch.full((g.num_nodes(),), -1, dtype=torch.long, device=device)
for _, row in df.iterrows():
    if row["protein_id"] in protein_to_idx:
        node_labels[protein_to_idx[row["protein_id"]]] = row["Enzyme"]

g.ndata["label"] = node_labels

# Extract Features and Labels
node_features = g.ndata['alphafold'].to(device)
# node_features = g.ndata['protvec'].to(device)
# node_features = g.ndata['biovec'].to(device)
# node_features = g.ndata['esm'].to(device)
# node_features = g.ndata['concatenated'].to(device)



labels = g.ndata['label'].to(device)


  
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.3):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, norm='both', weight=True, bias=True)
        self.conv3 = GraphConv(hidden_feats, out_feats, norm='both', weight=True, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, g, x):
        edge_weight = g.edata['weight'] if 'weight' in g.edata else None

        x = F.relu(self.conv1(g, x, edge_weight=edge_weight))
        x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(g, x, edge_weight=edge_weight))
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.conv3(g, x, edge_weight=edge_weight)
        x = self.layer_norm(x)
        return x

# Define Contrastive Loss with CosineEmbeddingLoss
def compute_loss(model, g, node_features):
    src, dst = g.edges()
    src, dst = src.to(device), dst.to(device)
    pos_embed = model(g, node_features)[src]
    neg_dst = torch.randint(0, g.num_nodes(), (len(dst),), device=device)
    neg_embed = model(g, node_features)[neg_dst]

    pos_labels = torch.ones(pos_embed.size(0), device=device)
    neg_labels = -torch.ones(neg_embed.size(0), device=device)

    criterion = nn.CosineEmbeddingLoss(margin=0.7)
    pos_loss = criterion(pos_embed, pos_embed, pos_labels)
    neg_loss = criterion(pos_embed, neg_embed, neg_labels)

    return pos_loss + neg_loss

# Model Training
hidden_dim = 64#64
embedding_dim = 32 #32

# in_feats = 384 (alphafold)
# in_feats = 300 (protvec)
# in_feats = 1024 (biovec)
# in_feats = 1280 (esm)
# in_feats = 2988 (concatenated)
model = GCN(in_feats=384, hidden_feats=hidden_dim, out_feats=embedding_dim).to(device)


# model = GraphSAGE(in_feats=384, hidden_feats=hidden_dim, out_feats=embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 200
for epoch in range(epochs):
    model.train()
    loss = compute_loss(model, g, node_features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.10f}")

# Compute Final Embeddings
model.eval()
node_embeddings = model(g, node_features).detach().cpu().numpy()

# =============================================================================
# # Train Logistic Regression for Classification
# =============================================================================

valid_indices = np.where(labels.cpu().numpy() >= 0)[0]
train_idx, test_idx, train_labels, test_labels = train_test_split(valid_indices, labels.cpu().numpy()[valid_indices], test_size=0.2, stratify=labels.cpu().numpy()[valid_indices])
train_embeddings, test_embeddings = node_embeddings[train_idx], node_embeddings[test_idx]
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)




clf = LogisticRegression(class_weight="balanced", max_iter=500)
clf.fit(train_embeddings_scaled, train_labels)

# Evaluate Model
y_pred = clf.predict(test_embeddings_scaled)
accuracy = (y_pred == test_labels).mean()
conf_matrix = confusion_matrix(test_labels, y_pred)

print(f"Test Classification Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Compute precision, recall, and F1-score
precision = precision_score(test_labels, y_pred, average='weighted')  # Weighted for class imbalance
recall = recall_score(test_labels, y_pred, average='weighted')
f1 = f1_score(test_labels, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")


