#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 11:31:23 2025
@author: saiful
"""

import os
import sys
import torch
import dgl
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import seaborn as sns

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Redirect output to file
# sys.stdout = open("console_outputs/console_output_gcn_kl.txt", "w")

# ---------------------------
# 1️⃣ Load Graph Data
# ---------------------------
with open("/data/saiful/ePPI/alphafold_ePPI_graph_with_map.pkl", "rb") as f:
    data = pickle.load(f)

g = data["graph"]  # DGL Graph
protein_to_idx = data["protein_to_idx"]  # Mapping protein_id → node index
g = dgl.add_self_loop(g)  # Add self-loops to graph

# Normalize node features
g.ndata["alphafold_feat"] = normalize(g.ndata["alphafold_feat"], p=2, dim=1)

print("Graph Loaded! Nodes:", g.num_nodes(), "Edges:", g.num_edges())

# ---------------------------
# 2️⃣ Load Labels and Assign to Graph
# ---------------------------
df = pd.read_csv("/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_enzym_info_all.csv")

df["Enzyme"] = df["Enzyme"].astype(int)  # Convert labels to binary (1=Enzyme, 0=Non-Enzyme)
node_labels = torch.full((g.num_nodes(),), -1, dtype=torch.long)  # Initialize with -1 (unlabeled)

# Assign labels to nodes
for _, row in df.iterrows():
    protein_id = row["protein_id"]
    if protein_id in protein_to_idx:
        node_labels[protein_to_idx[protein_id]] = row["Enzyme"]

g.ndata["label"] = node_labels  # Attach labels to graph
features = g.ndata["alphafold_feat"]  # Node feature matrix (shape: [num_nodes, 384])
labels = g.ndata["label"]  # Label tensor

print("Label Distribution:", torch.bincount(node_labels[node_labels >= 0]))

# ---------------------------
# 3️⃣ Define GraphSAGE Model
# ---------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, "mean")
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, "mean")
        self.conv3 = SAGEConv(hidden_feats, out_feats, "mean")

        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.bn1(x)
        x = self.dropout(x)

        x_res = x
        x = F.relu(self.conv2(g, x))
        x = x + x_res  # Residual connection
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.conv3(g, x)
        x = self.layer_norm(x)
        return x

# ---------------------------
# 4️⃣ Define KL Divergence Loss (Fixed for NaN)
# ---------------------------
class KLDivergenceLoss(nn.Module):
    def __init__(self, lambda_kl=0.5):
        super(KLDivergenceLoss, self).__init__()
        self.lambda_kl = lambda_kl  # Weight for KL Divergence loss

    def forward(self, predictions, labels, soft_targets):
        labeled_mask = labels >= 0
        unlabeled_mask = labels == -1

        # Compute standard cross-entropy loss on labeled nodes
        ce_loss = F.cross_entropy(predictions[labeled_mask], labels[labeled_mask])

        # Prevent log(0) error by adding small epsilon
        kl_loss = F.kl_div(F.log_softmax(predictions[unlabeled_mask] + 1e-2, dim=-1),
                           soft_targets[unlabeled_mask], reduction="batchmean")

        # Combine losses
        loss = ce_loss + self.lambda_kl * kl_loss
        return loss

# ---------------------------
# 5️⃣ Training Loop
# ---------------------------
def compute_loss(model, g, features, labels):
    predictions = model(g, features)

    # Compute soft targets using softmax (pseudo-labels for unlabeled nodes)
    soft_targets = F.softmax(predictions.detach(), dim=-1)

    # Compute KL Divergence Loss
    loss = KLDivergenceLoss(lambda_kl=0.5)(predictions, labels, soft_targets)
    return loss

# Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_feats=384, hidden_feats=64, out_feats=2).to(device)  # Out_feats=2 for classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Move data to GPU
g = g.to(device)
features = features.to(device)
labels = labels.to(device)

# Train the model
epochs = 500
for epoch in range(epochs):
    model.train()
    loss = compute_loss(model, g, features, labels)

    optimizer.zero_grad()
    loss.backward()

    # Apply gradient clipping to avoid NaN loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: KL Divergence Loss = {loss.item():.4f}")

# ---------------------------
# 6️⃣ Compute Node Embeddings
# ---------------------------
model.eval()
node_embeddings = model(g, features).detach().cpu().numpy()

# ---------------------------
# 7️⃣ Train Logistic Regression for Classification
# ---------------------------
labeled_mask = labels.cpu().numpy() >= 0
train_idx, test_idx, train_labels, test_labels = train_test_split(
    np.where(labeled_mask)[0], labels.cpu().numpy()[labeled_mask], test_size=0.2, stratify=labels.cpu().numpy()[labeled_mask]
)

train_embeddings, test_embeddings = node_embeddings[train_idx], node_embeddings[test_idx]

# Train classifier
clf = LogisticRegression(class_weight="balanced", max_iter=500)
clf.fit(train_embeddings, train_labels)

# ---------------------------
# 8️⃣ Evaluate Model Performance
# ---------------------------
y_pred = clf.predict(test_embeddings)
accuracy = accuracy_score(test_labels, y_pred)
print(f"Test Classification Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", conf_matrix)


