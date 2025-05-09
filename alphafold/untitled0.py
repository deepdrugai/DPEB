#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:29:26 2024

@author: saiful
"""

import pickle

# Load the pickle file
with open('/home/saiful/ePPI_dgl/alphafold/results/embeddings/individual_embeddings2/A0A023GRW2_embedding.pkl', 'rb') as handle:
    protein_embeddings = pickle.load(handle)

embedding =protein_embeddings['embedding']
print(f"Embedding shape: {embedding.shape}")

#%%
# Load the pickle file
with open('/home/saiful/ePPI_dgl/alphafold/results/embeddings/individual_embeddings/A0A024QZ98_embedding.pkl', 'rb') as handle:
    protein_embeddings2 = pickle.load(handle)

embedding2 =protein_embeddings2['embedding']
print(f"Embedding shape: {embedding2.shape}")


# "/data/saiful/ePPI/prot_to_fasta_all.txt"

#%%

import pandas as pd


# Load the data from the provided file
file_path = "/home/saiful/ePPI_dgl/alphafold/eppi_sequences.txt"
df = pd.read_csv(file_path, delimiter='\t', header=None)

# Create a function to convert the dataframe to FASTA format
def convert_to_fasta(df):
    fasta_format = ""
    for index, row in df.iterrows():
        fasta_format += f">{row[0]} \n{row[1]}\n"
        # fasta_format += f">{row[0]} {row[1]}\n"
    return fasta_format

# Convert the dataframe to FASTA format
fasta_output = convert_to_fasta(df)

# Write the output to a new file
output_file_path = "/home/saiful/ePPI_dgl/alphafold/sequences_converted.txt"
with open(output_file_path, 'w') as f:
    f.write(fasta_output)

# Output the path of the newly created file
print(f"The FASTA formatted file has been saved to: {output_file_path}")
