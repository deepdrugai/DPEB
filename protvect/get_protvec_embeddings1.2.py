#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:52:44 2025

@author: saiful
"""
# use conda environment ePPI4
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:34:50 2025
@author: saiful
"""
#%% cell 2 
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Load ProtVec model
model = Word2Vec.load('/home/saiful/ePPI_dgl/protvec/swissprot-reviewed-protvec.model')

# Output folder
output_dir = "/data/saiful/ePPI/other_embeddings/protvec_embeddings/protvec_dict_embeddings"
os.makedirs(output_dir, exist_ok=True)

def tokenize_protein_offset(protein_sequence, k=3, offset=0):
    return [protein_sequence[i:i+k] for i in range(offset, len(protein_sequence) - k + 1, 3)]


def get_raw_protein_embeddings(protein_sequence, model, k=3):
    all_embeddings = []

    # Loop over 3 possible reading frame offsets
    for offset in range(3):
        kmers = tokenize_protein_offset(protein_sequence, k, offset)

        # Collect embeddings only for valid k-mers present in the model
        for kmer in kmers:
            if kmer in model.wv:
                all_embeddings.append(model.wv[kmer])

    # If no valid k-mer was found, return a fallback zero vector
    if not all_embeddings:
        return np.zeros((1, model.vector_size))  # Shape: (1, 100)

    return np.array(all_embeddings, dtype=np.float32)  # Shape: (N, 100)


def save_protvec_embedding_dicts(prot_to_fasta_df, model):
    total_embeddings = 0  # Counter for total number of k-mer embeddings
    
    for idx, row in prot_to_fasta_df.iterrows():
        protein_id = row[0]
        fasta = row[1]
        try:
            embedding = get_raw_protein_embeddings(fasta, model)
            total_embeddings += embedding.shape[0]  # Count number of rows (k-mers)
            
            prot_dict = {
                'protein_id': protein_id,
                'fasta': fasta,
                'embedding': embedding
            }
            save_path = os.path.join(output_dir, f"{protein_id}_protvec.npy")
            np.save(save_path, prot_dict, allow_pickle=True)
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
    
    print(f"\n Total number of k-mer embeddings generated across all proteins: {total_embeddings}")


# Load 23K fasta dataset and use first 50 as an example
file_path = "/data/saiful/ePPI/prot_to_fasta_23k.csv"
prot_to_fasta_23k = pd.read_csv(file_path)

# prot_to_fasta_23k = pd.read_csv(file_path).iloc[:50]  # Limit for test
save_protvec_embedding_dicts(prot_to_fasta_23k, model)


#%% Code to Select 5 Random Files and Check Shapes
    
import os
import numpy as np
import random

# Directory path
folder_path = "/data/saiful/ePPI/other_embeddings/protvec_embeddings/protvec_dict_embeddings"

# List all .npy files
npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# Randomly sample 5 files
random_files = random.sample(npy_files, 5)

# Load and print embedding shape for each
for file_name in random_files:
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path, allow_pickle=True).item()
    
    protein_id = data['protein_id']
    embedding = data['embedding']
    
    print(f"File: {file_name}")
    print(f"Protein ID: {protein_id}")
    print(f"Embedding shape: {embedding.shape}\n")

#%%
import numpy as np

# Path to the file you want to load
file_path = "/home/saiful/ePPI_dgl/protvec/protvec_dict_embeddings/A0A024R0L9_protvec.npy"

# Load the dictionary (note allow_pickle=True is required)
protein_dict = np.load(file_path, allow_pickle=True).item()

# Inspect contents
print(f"Protein ID: {protein_dict['protein_id']}")
print(f"FASTA length: {len(protein_dict['fasta'])}")
print(f"Embedding shape: {protein_dict['embedding'].shape}")
print(f"First row of embedding:\n{protein_dict['embedding'][0]}")

#%%
import os

# Path to your embeddings directory
folder_path = "/data/saiful/ePPI/other_embeddings/protvec_embeddings/protvec_dict_embeddings/"

# List all files ending with .npy
npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# Count and print
print(f"Total number of .npy embedding files: {len(npy_files)}")

