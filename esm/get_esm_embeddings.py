#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 18:56:15 2025

@author: saiful
"""
# /home/saiful/anaconda3/envs/ePPI_dgl38_esm/bin/esm-extract esm2_t33_650M_UR50D \
#   /home/saiful/ePPI_dgl/esm/esm2_input.fasta \
#   /data/saiful/ePPI/other_embeddings/esm_embeddings/esm_pt_files/ \
#   --repr_layers 33 \
#   --include mean per_tok \

# # use the environment ePPI_dgl38_esm
import pandas as pd

# Load first 50 proteins
csv_path = "/data/saiful/ePPI/prot_to_fasta_23k.csv"
df = pd.read_csv(csv_path)

# df = pd.read_csv(csv_path).iloc[:50]

# Save to FASTA
fasta_path = "./esm2_input.fasta"
with open(fasta_path, "w") as fasta_file:
    for _, row in df.iterrows():
        protein_id = row[0]
        sequence = row[1]
        fasta_file.write(f">{protein_id}\n{sequence}\n")

print(f" FASTA saved to {fasta_path}")
import os

pt_dir = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm_pt_files"

# Count .pt files
pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
print(f"Total number of .pt files: {len(pt_files)}")
#%%
import os
import torch
import numpy as np
import pandas as pd

# === CONFIG ===
csv_path = "/data/saiful/ePPI/prot_to_fasta_23k.csv"
fasta_path = "/home/saiful/ePPI_dgl/esm/esm2_input.fasta"  # Generated from your .csv
esm2_embedding_dir = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm_pt_files" # Folder with .pt files from extract.py
save_dir = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm2_dict_embeddings"
EMB_LAYER = 33  # Layer to extract from

os.makedirs(save_dir, exist_ok=True)

# === Load first 50 proteins ===
# df = pd.read_csv(csv_path).iloc[:50]
print("Loaded fasta file:", csv_path)
print("Total proteins to process:", len(df))

# === Create protein_id to fasta dict ===
protein_to_fasta = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# === Process and save each protein ===
success_count = 0
missing_files = []

for protein_id, fasta in protein_to_fasta.items():
    try:
        # Look for matching .pt file based on header prefix
        pt_files = [f for f in os.listdir(esm2_embedding_dir) if f.startswith(protein_id) and f.endswith(".pt")]
        if not pt_files:
            missing_files.append(protein_id)
            continue

        # Load the .pt file
        pt_path = os.path.join(esm2_embedding_dir, pt_files[0])
        emb = torch.load(pt_path)

        # Get full sequence embedding (L, D)
        embedding = emb["representations"][EMB_LAYER]  # shape: (L, 1280)

        # Save dictionary
        prot_dict = {
            "protein_id": protein_id,
            "fasta": fasta,
            "embedding": embedding.numpy()
        }

        save_path = os.path.join(save_dir, f"{protein_id}_esm2.npy")
        np.save(save_path, prot_dict, allow_pickle=True)
        success_count += 1

    except Exception as e:
        print(f"Error with {protein_id}: {e}")
        missing_files.append(protein_id)

print(f"\n Successfully saved embeddings for {success_count} proteins.")
if missing_files:
    print(" Missing or failed proteins:", missing_files)

#%% checking a sample pt file
import torch
"/data/saiful/ePPI/other_embeddings/esm_embeddings/esm_pt_files/A0A0G2JQH2.pt"
pt_path = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm_pt_files/A0A0G2JQH2.pt"
d = torch.load(pt_path)

print("Available keys:", d.keys())
#%%
import numpy as np

# File path
file_path = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm2_dict_embeddings/A0A024QZS4_esm2.npy"

# Load the dictionary
data = np.load(file_path, allow_pickle=True).item()

# Print keys
print("Top-level keys in the dictionary:", data.keys())

# Optional: Inspect individual parts
print("Protein ID:", data.get('protein_id'))
print("FASTA length:", len(data.get('fasta', '')))
print("Embedding shape:", data['embedding'].shape)

#%%
import os
import numpy as np

# Directory containing .npy files
dir_path = "/data/saiful/ePPI/other_embeddings/esm_embeddings/esm2_dict_embeddings"

# List all .npy files
npy_files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]

# Iterate and load each file
for fname in npy_files:
    fpath = os.path.join(dir_path, fname)
    try:
        data = np.load(fpath, allow_pickle=True).item()
        protein_id = data.get('protein_id', 'UNKNOWN')
        embedding_shape = data['embedding'].shape
        print(f"{fname} | Protein ID: {protein_id} | Embedding shape: {embedding_shape}")
    except Exception as e:
        print(f"❌ Error reading {fname}: {e}")
