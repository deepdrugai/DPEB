#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:07:47 2025

@author: saiful
"""

import pandas as pd

file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
df = pd.read_csv(file_path)
print(df.head())

df.columns

# Drop the specified columns
df.drop(columns=['FASTA', 'biovec_UniProtID', 'alphafold_protein_id', 'esm_ProteinID'], inplace=True)
df.rename(columns={'protvec_protein_id': 'protein_id'}, inplace=True)

print(df.columns)

#%%  df_new contains all the edge info
import pandas as pd
df_new = pd.read_csv("/data/saiful/ePPI/df_new.csv", index_col=0)
df_new = df_new[df_new.isHuman == 1]
df_new = df_new.reset_index(drop=True)

print(df_new.head())
weights_from_df_new = df_new['weight'].tolist()  # 11799878

df_edge = df_new

#%%

# Extract the 'protein_id' column into a list
protein_id_list = df['protein_id'].tolist()

import pandas as pd
import dgl 
import torch
# remove rows where either source or target protein  is not present from protein_ids list.
df_filtered = df_new[
    df_new['source'].isin(protein_id_list) & df_new['target'].isin(protein_id_list)
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

import pandas as pd
import dgl
import torch
import ast
import numpy as np

# Assuming your existing code has already created `g`, `df`, and `protein_to_idx`
# g = dgl.graph((src_indices, tgt_indices), num_nodes=len(all_proteins))
# protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}

# Step 1: Prepare the embedding columns from df
embedding_types = ['protvec_embedding', 'biovec_Embeddings', 'alphafold_embeddings', 'esm_embeddings']

# Step 2: Convert string embeddings to lists (if stored as strings in CSV)
for emb_type in embedding_types:
    # Check if the column contains string representations of lists
    if isinstance(df[emb_type].iloc[0], str):
        df[emb_type] = df[emb_type].apply(ast.literal_eval)  # Safely convert string to list

# Step 3: Create a mapping from protein_id to embeddings
protein_embedding_dict = {row['protein_id']: {emb_type: row[emb_type] for emb_type in embedding_types} 
                          for _, row in df.iterrows()}

# Step 4: Initialize node feature tensors for each embedding type
# Get the length of each embedding type from the first row (assuming consistent sizes)
sample_protein = df['protein_id'].iloc[0]
embedding_sizes = {emb_type: len(protein_embedding_dict[sample_protein][emb_type]) 
                   for emb_type in embedding_types}

# Initialize empty tensors for all nodes, filled with zeros initially
num_nodes = g.num_nodes()
node_features = {
    'protvec': torch.zeros((num_nodes, embedding_sizes['protvec_embedding']), dtype=torch.float32),
    'biovec': torch.zeros((num_nodes, embedding_sizes['biovec_Embeddings']), dtype=torch.float32),
    'alphafold': torch.zeros((num_nodes, embedding_sizes['alphafold_embeddings']), dtype=torch.float32),
    'esm': torch.zeros((num_nodes, embedding_sizes['esm_embeddings']), dtype=torch.float32)
}

# Step 5: Populate node features using protein_to_idx
for protein, idx in protein_to_idx.items():
    if protein in protein_embedding_dict:
        # Assign embeddings to the corresponding node index
        node_features['protvec'][idx] = torch.tensor(protein_embedding_dict[protein]['protvec_embedding'], dtype=torch.float32)
        node_features['biovec'][idx] = torch.tensor(protein_embedding_dict[protein]['biovec_Embeddings'], dtype=torch.float32)
        node_features['alphafold'][idx] = torch.tensor(protein_embedding_dict[protein]['alphafold_embeddings'], dtype=torch.float32)
        node_features['esm'][idx] = torch.tensor(protein_embedding_dict[protein]['esm_embeddings'], dtype=torch.float32)
    else:
        # If a protein in the graph isn’t in df, it keeps the zero-initialized embedding
        print(f"Warning: Protein {protein} not found in df, using zero embedding.")

# Step 6: Assign node features to the graph
g.ndata['protvec'] = node_features['protvec']
g.ndata['biovec'] = node_features['biovec']
g.ndata['alphafold'] = node_features['alphafold']
g.ndata['esm'] = node_features['esm']

# Step 7: Verify
print("Graph with node features:")
print(g)
print("Node feature shapes:")
for feat_name in g.ndata:
    print(f"{feat_name}: {g.ndata[feat_name].shape}")

# # Optional: Save the protein_to_idx mapping for later use
# import pickle
# with open('protein_to_idx.pkl', 'wb') as f:
#     pickle.dump(protein_to_idx, f)

# # You can also save the graph if needed
# dgl.save_graphs('protein_graph.dgl', [g])


import pickle 
save_path = "/data/saiful/ePPI/all_ePPI_embeddings_graph_with_map.pkl"

# Save both the graph and the protein_to_idx dictionary
with open(save_path, "wb") as f:
    pickle.dump({"graph": g, "protein_to_idx": protein_to_idx}, f)
    
print("Graph saved in both DGL and Pickle formats.")

#%% loading the graph with mapping 

with open("/data/saiful/ePPI/all_ePPI_embeddings_graph_with_map.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the graph and mapping
g1 = data["graph"]
protein_to_idx1 = data["protein_to_idx"]

print("Graph and protein mapping loaded successfully!")
print("Number of nodes in graph:", g1.num_nodes())
print("Example mapping:", list(protein_to_idx1.items())[:5])  # Print first 5 mappings

print(g1.edata['weight'][:10])


feature_key = 'alphafold'

# Loop through a few proteins from the mapping
for protein, idx in list(protein_to_idx1.items())[:5]:
    node_feature = g1.ndata[feature_key][idx]
    print(f"Protein: {protein}, {feature_key} feature: {node_feature[:5]}")


















