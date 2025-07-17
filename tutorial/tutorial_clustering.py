#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:47:09 2025

@author: saiful
"""

#%% aggregate the alphafold embeddings using mean pooling and save them as a csv file
import os
import pickle
import numpy as np
import pandas as pd

# Folder containing embeddings
embedding_folder = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np_v1.3/"

# Load embeddings into a dictionary
embeddings = {}
for file in os.listdir(embedding_folder):
    if file.endswith(".npy"):
        
        file_path = os.path.join(embedding_folder, file)
        data = np.load(file_path, allow_pickle=True).item()  

        protein_id = data['protein_id'] 
        fasta = data['fasta']  
        embedding = data['embedding']
        embeddings[protein_id] = {'fasta': fasta, 'embedding': embedding}
     
# Aggregate embeddings using mean pooling

aggregated_embeddings = {}
for protein_id, protein_data in embeddings.items():
    fasta = protein_data['fasta']  
    mean_embedding = np.mean(protein_data['embedding'], axis=0)  # Shape: (380,)
    aggregated_embeddings[protein_id] = {'fasta': fasta, 'embedding': mean_embedding}
    
# aggregated_embeddings.keys()= 'Q8NHW5', 'Q8N428', 'Q8NI32'])
# aggregated_embeddings['Q8NHW5'].keys() = dict_keys(['fasta', 'embedding'])


# Print only the first 5 entries from the aggregated_embeddings dictionary
for i, (protein_id, data) in enumerate(aggregated_embeddings.items()):
    if i >= 5:
        break
    print(f"Protein ID: {protein_id}")
    print(f"FASTA: {data['fasta']}")
    print(f"Aggregated Embedding Shape: {data['embedding'].shape}")
    print("Aggregated Embedding Values:")
    print(data['embedding'])  # Print the actual mean embedding vector
    print()



# Convert the aggregated embeddings into a DataFrame
data = []
for protein_id, protein_data in aggregated_embeddings.items():
    row = {
        'protein_id': protein_id,
        'fasta': protein_data['fasta'],
        'aggregated_features': protein_data['embedding'].tolist()  # Convert numpy array to list
    }
    data.append(row)

df_alphafold_aggregated = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_csv_path = "/data/saiful/ePPI/tutorial/eppi_alphafold_aggregated_embeddings.csv"
df_alphafold_aggregated.to_csv(output_csv_path, index=False)

print(f"Aggregated embeddings saved to {output_csv_path}")

#%%

import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ast
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
embed_type = 'Alphafold'
import sys
# sys.stdout = open(f"console_outputs/console_output_{embed_type}2.txt", "w")
# sys.stdout = open("console_outputs/console_output", "w")

print(f"Embedding type: {embed_type}")

#%% Load and Preprocess Data
def load_and_filter_data(protein_file, embedding_file):
    try:
        # Load protein families
        df_protein_families = pd.read_csv(protein_file)
        df_selected = df_protein_families[['Protein_ID', 'Family']]
        df_filtered = df_selected[~df_selected['Family'].isin(["No SIMILARITY comment found", "No family label found"])]
        
        # Filter families with >60 occurrences
        family_frequencies = df_filtered['Family'].value_counts()
        frequent_families = family_frequencies[family_frequencies > 60].index
        df_filtered = df_filtered[df_filtered['Family'].isin(frequent_families)]
        
        # Load AlphaFold embeddings
        df_alphafold = pd.read_csv(embedding_file)
        df_alphafold['aggregated_features'] = df_alphafold['aggregated_features'].apply(ast.literal_eval)
        
        # Merge dataframes
        merged_df = pd.merge(df_alphafold, df_filtered, left_on='protein_id', right_on='Protein_ID', how='inner')
        
        print(f"Merged DataFrame shape: {merged_df.shape}")
        print(f"Unique families: {df_filtered['Family'].nunique()}")
        return merged_df
    except Exception as e:
        print(f"Error in load_and_filter_data: {e}")
        raise

# Load data
protein_file = "/home/saiful/ePPI_dgl/clustering/protein_families23k.csv"
embedding_file = output_csv_path  # "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
try:
    merged_df = load_and_filter_data(protein_file, embedding_file)
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

embeddings_raw = np.array(merged_df['aggregated_features'].tolist())
true_labels = merged_df['Family']
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)
print(f"Raw {embed_type} Embeddings shape: {embeddings_raw.shape}")
print(f"Number of unique labels: {len(np.unique(true_labels_encoded))}")

#%% Plot Raw Embeddings
def plot_embeddings(embeddings, labels, title, save_path, label_type="Class"):
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='tab20', s=20, alpha=0.8)
        plt.colorbar(scatter, label=label_type)
        plt.title(title)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in plot_embeddings: {e}")
        raise

# Plot raw embeddings with true labels
plot_embeddings(embeddings_raw, true_labels_encoded, f"t-SNE Visualization of Raw {embed_type} Embeddings", 
                f"/home/saiful/ePPI_dgl/clustering/figures_paper/raw_embeddings_tsne_{embed_type}.png")

#%% K-means Clustering on Raw Embeddings
def evaluate_kmeans(embeddings, true_labels_encoded, label_encoder, title, save_path):
    try:
        from sklearn.decomposition import PCA

        # Apply PCA to retain 95% of the variance
        pca = PCA(n_components=0.95, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"PCA reduced shape for {title}: {embeddings_reduced.shape}")
        
        # Perform K-means clustering on PCA-reduced embeddings
        k = len(np.unique(true_labels_encoded))
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_reduced)
        
        # Map cluster labels to true labels via majority voting
        cluster_to_label = {}
        for cluster in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster)[0]
            true_labels_cluster = true_labels_encoded[indices]
            majority_label = np.bincount(true_labels_cluster).argmax()
            cluster_to_label[cluster] = majority_label
        
        mapped_labels = np.array([cluster_to_label[cluster] for cluster in cluster_labels])
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_labels_encoded, mapped_labels),
            'precision': precision_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0),
            'recall': recall_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels_encoded, mapped_labels, average='weighted', zero_division=0)
        }
        
        print(f"\n============K-means Clustering on {title}==========")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        # print("\nDetailed Classification Report:")
        # print(classification_report(true_labels_encoded, mapped_labels, target_names=label_encoder.classes_, zero_division=0))
        
        # Visualize clusters using t-SNE on PCA-reduced embeddings
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings_reduced)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab20', s=20, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"t-SNE Visualization of {title} with K-means Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.show()
        
        return metrics
    except Exception as e:
        print(f"Error in evaluate_kmeans ({title}): {e}")
        return None
    
# Evaluate K-means on raw embeddings
kmeans_raw_metrics = evaluate_kmeans(embeddings_raw, true_labels_encoded, label_encoder, 
                                    f"Raw {embed_type} Embeddings", 
                                    f"/home/saiful/ePPI_dgl/clustering/figures_paper/raw_embeddings_kmeans_tsne_{embed_type}.png")

