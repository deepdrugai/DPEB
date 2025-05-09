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
    protein_embeddings = pickle.load(handle)

embedding2 =protein_embeddings['embedding']
print(f"Embedding shape: {embedding2.shape}")


"/data/saiful/ePPI/prot_to_fasta_all.txt"

#%% Input File (prot_to_fasta_23k.txt):  ==> Output File (prot_to_fasta_23k_alphafold.txt):

import pandas as pd


# Load the data from the provided file
# file_path = "/home/saiful/ePPI_dgl/alphafold/sequences.txt"

file_path = "/data/saiful/ePPI/prot_to_fasta_23k.txt"



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
# output_file_path = "/home/saiful/ePPI_dgl/alphafold/sequences_converted.txt"

output_file_path = "/data/saiful/ePPI/prot_to_fasta_23k_alphafold.txt"

with open(output_file_path, 'w') as f:
    f.write(fasta_output)

# Output the path of the newly created file
print(f"The FASTA formatted file has been saved to: {output_file_path}")

#%% split the 23k alphafold files splits it into smaller files (each containing 500 sequences), and saves them in a specified directory


import os

# Define the directory and input file paths
input_file_path = '/data/saiful/ePPI/prot_to_fasta_23k_alphafold.txt'
output_dir = '/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the data from the input file
with open(input_file_path, 'r') as f:
    lines = f.readlines()

# Initialize variables for splitting
batch_size = 500  # 500 sequences per file
batch_count = 1
batch = []
sequence_count = 0

# Loop through the file and split into batches of 500 sequences
for line in lines:
    batch.append(line)
    if line.startswith('>'):
        sequence_count += 1

        # When 500 sequences are collected, write to a new file
        if sequence_count == batch_size:
            output_file_path = os.path.join(output_dir, f'batch_{batch_count}.txt')
            with open(output_file_path, 'w') as out_f:
                out_f.writelines(batch)
            batch_count += 1
            batch = []  # Reset for the next batch
            sequence_count = 0

# Save any remaining sequences in the last batch
if batch:
    output_file_path = os.path.join(output_dir, f'batch_{batch_count}.txt')
    with open(output_file_path, 'w') as out_f:
        out_f.writelines(batch)

print(f"Data has been split into {batch_count} batches and saved to {output_dir}")

#%%



