#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 03:07:33 2025

@author: saiful
"""

#%% aggregate the alphafold embeddings using mean pooling and save them as a csv file
import os
import pickle
import numpy as np
import pandas as pd


# =============================================================================
# #%%                     now make alphafold graph 
# =============================================================================
import os

folder_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/"
# List all files in the folder
files = os.listdir(folder_path)
protein_ids = [file.split('_')[0] for file in files if file.endswith('.pkl')]
print(protein_ids)

#%%  df_new contains all the edge info
import pandas as pd
df_new = pd.read_csv("/data/saiful/ePPI/df_new.csv", index_col=0)
df_new = df_new[df_new.isHuman == 1]
df_new = df_new.reset_index(drop=True)

print(df_new.head())
weights_from_df_new = df_new['weight'].tolist()  # 11799878

#%% remove rows from df_new where neither the source nor the target proteins are found in the protein_ids list

import pandas as pd
import dgl 
import torch
# remove rows where either source or target protein  is not present from protein_ids list.
df_filtered = df_new[
    df_new['source'].isin(protein_ids) & df_new['target'].isin(protein_ids)
]
# Reset the index of the filtered DataFrame
df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered.head())
       
#%% =============================================================================
# ##   == ==   convert df_filtered into a DGL graph  == ==  ##
# =============================================================================
import ast
#  Create unique node mapping
all_proteins = set(df_filtered["source"]).union(set(df_filtered["target"]))
protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}

#  Convert protein IDs to numerical indices
src_indices = df_filtered["source"].map(protein_to_idx).tolist()
tgt_indices = df_filtered["target"].map(protein_to_idx).tolist()

#  Create a DGL graph
g = dgl.graph((src_indices, tgt_indices), num_nodes=len(all_proteins))
#  Add edge weights
g.edata["weight"] = torch.tensor(df_filtered["weight"].values, dtype=torch.float32)
# g.edata["isHuman"] = torch.tensor(df_filtered["isHuman"].values, dtype=torch.int32)
print(g)
#%%
import pickle

# now loading node features
# this file contains protein id, seq and AF embeddings
file_path = "/home/saiful/ePPI_dgl/bioembedding/bio_embeddings_ePPI.csv"
df_id_seq_BE = pd.read_csv(file_path)
print("df_id_seq_BE:" ,df_id_seq_BE.head())

df_id_seq_BE = df_id_seq_BE.rename(columns={'UniProt ID': 'protein_id'})

# Filter df_id_seq_BE to keep only proteins that are in the graph
df_id_seq_BE_filtered = df_id_seq_BE[df_id_seq_BE["protein_id"].isin(all_proteins)]
#  Convert Embeddings from string to list
df_id_seq_BE_filtered["Embeddings"] = df_id_seq_BE_filtered["Embeddings"].apply(ast.literal_eval)

#  Create a tensor to store node features (initialize with zeros)
feature_dim = len(df_id_seq_BE_filtered.iloc[0]["Embeddings"])  
node_features = torch.zeros((len(all_proteins), feature_dim), dtype=torch.float32)

#  Assign embeddings to correct node index
for _, row in df_id_seq_BE_filtered.iterrows():
    protein = row["protein_id"]
    if protein in protein_to_idx:
        node_idx = protein_to_idx[protein]
        node_features[node_idx] = torch.tensor(row["Embeddings"], dtype=torch.float32)

#  Add node features to the DGL graph
g.ndata["bioembed_feat"] = node_features
g_alpha = g

# 9. Save the graph
# dgl.save_graphs("/data/saiful/ePPI/alphafold_ePPI_graph.dgl", g)

# # Save the same graph as a pickle file
# with open("/data/saiful/ePPI/alphafold_ePPI_graph.pkl", "wb") as f:
#     pickle.dump(g, f)
    
save_path = "/data/saiful/ePPI/bioembed_ePPI_graph_with_map.pkl"

# Save both the graph and the protein_to_idx dictionary
with open(save_path, "wb") as f:
    pickle.dump({"graph": g, "protein_to_idx": protein_to_idx}, f)
    
print("Graph saved in both DGL and Pickle formats.")


#%% check some values from the pkl file 

import pickle
# with open("/data/saiful/ePPI/prot_esm_ePPI_graph_test.pkl", 'rb') as f:
#   g = pickle.load(f)
  
with open("/data/saiful/ePPI/bioembed_ePPI_graph_with_map.pkl", 'rb') as f:
  g = pickle.load(f)
            
node_index = 22  
print(g.ndata["bioembed_feat"][node_index])

#%% loading the graph with mapping 

with open("/data/saiful/ePPI/bioembed_ePPI_graph_with_map.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the graph and mapping
g = data["graph"]
protein_to_idx = data["protein_to_idx"]

print("Graph and protein mapping loaded successfully!")
print("Number of nodes in graph:", g.num_nodes())
print("Example mapping:", list(protein_to_idx.items())[:5])  # Print first 5 mappings

# Retrieve Node Features for the First 5 Proteins

import torch

# Get the indices of the first 5 proteins
protein_ids = ['Q68DU8', 'P04279', 'P51800', 'P13591', 'Q16647']
node_indices = [protein_to_idx[pid] for pid in protein_ids]  # Convert protein IDs to indices

# Extract node features for these indices
node_features = g.ndata["bioembed_feat"][node_indices]  # Assuming 'feat' is the feature key

# Print node feature values
for pid, idx, feature in zip(protein_ids, node_indices, node_features):
    print(f"Protein: {pid} | Node Index: {idx} | Feature Vector: {feature.tolist()}")


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            