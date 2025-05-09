#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:09:54 2025

@author: saiful
"""

import torch
import dgl
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import SpectralEmbedding
import pandas as pd

# Load Graph
with open("/data/saiful/ePPI/alphafold_ePPI_graph_with_map.pkl", "rb") as f:
    data = pickle.load(f)

g = data["graph"]  # DGL Graph
protein_to_idx = data["protein_to_idx"]

# Extract AlphaFold Features (Node Features)
alpha_fold_features = g.ndata["alphafold_feat"]  # Shape: (21858, 384)

# Normalize Features
scaler = StandardScaler()
alpha_fold_features = scaler.fit_transform(alpha_fold_features.numpy())

# Compute Spectral Embeddings
embedding_dim = 2  # Adjust as needed
spectral_embedding = SpectralEmbedding(n_components=embedding_dim, affinity="nearest_neighbors")
node_embeddings = spectral_embedding.fit_transform(alpha_fold_features)

# Perform Clustering (K-Means)
num_clusters = 10  # Adjust based on your dataset
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(node_embeddings)

# t-SNE Visualization for Clusters
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(node_embeddings)

# Plot Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=clusters, palette="viridis", s=100, alpha=0.8)
plt.title("Spectral Clustering of Proteins using Graph Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.show()

# Evaluate Clustering Performance
silhouette_avg = silhouette_score(node_embeddings, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Save Clustering Results
protein_list = list(protein_to_idx.keys())
cluster_df = pd.DataFrame({"Protein": protein_list, "Cluster": clusters})
# cluster_df.to_csv("protein_clusters.csv", index=False)

print("Clustering results saved to protein_clusters.csv")
