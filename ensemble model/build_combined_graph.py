#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 06:27:59 2025

@author: saiful
"""
#%% # load protvec df
                            

import pandas as pd 
file_path = "/data/saiful/ePPI/protvec_embeddings_eppi.csv"

df_protvec = pd.read_csv(file_path)
print("df_protvec:" ,df_protvec.head())
# Rename columns using a dictionary
df_protvec.rename(columns={
    'protein_id': 'protvec_protein_id',
    'fasta': 'protvec_fasta',
    'embedding': 'protvec_embedding'
}, inplace=True)

# Check the updated column names
print(df_protvec.columns)

#%%# load biovec df 

import pandas as pd 
file_path = "/home/saiful/ePPI_dgl/bioembedding/bio_embeddings_ePPI.csv"
df_biovec = pd.read_csv(file_path)
print("df_biovec:" ,df_biovec.head())

# Rename columns using a dictionary
df_biovec.rename(columns={
    'UniProt ID': 'biovec_UniProtID',
    'FASTA Sequence': 'biovec_FASTA Sequence',
    'Embedding Shape': 'biovec_Embedding Shape',
    'Embeddings': 'biovec_Embeddings'
}, inplace=True)

# Check the updated column names
print(df_biovec.columns)

#%% # load alphafold  df 
                            

import pandas as pd 
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"

df_alphafold = pd.read_csv(file_path)
print("df_alphafold:" ,df_alphafold.head())


# Rename columns using a dictionary
df_alphafold.rename(columns={
    'protein_id': 'alphafold_protein_id',
    'fasta': 'alphafold_fasta',
    'aggregated_features': 'alphafold_aggregated_features'
}, inplace=True)

# Check the updated column names
print(df_alphafold.columns)

#%% # load esm  df 

file_path = "/data/saiful/ePPI/ProteinID_proteinSEQ_ESM_emb.csv"

df_esm = pd.read_csv(file_path)
print("df_esm:" ,df_esm.head())
# Rename columns using a dictionary
df_esm.rename(columns={
    'ProteinID': 'esm_ProteinID',
    'Protein_sequence': 'esm_Protein_sequence',
    'ESM_Embeddings': 'esm_ESM_Embeddings'
}, inplace=True)

# Check the updated column names
print(df_esm.columns)

# =============================================================================
#%%
import pandas as pd

# Merge all four DataFrames on common protein IDs
merged_df = pd.merge(
    df_protvec, 
    df_biovec, 
    left_on='protvec_protein_id', 
    right_on='biovec_UniProtID', 
    how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df_alphafold, 
    left_on='protvec_protein_id', 
    right_on='alphafold_protein_id', 
    how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df_esm, 
    left_on='protvec_protein_id', 
    right_on='esm_ProteinID', 
    how='inner'
)

# Drop duplicate columns if needed (e.g., multiple ID columns from merging)
# merged_df.drop(columns=['biovec_UniProtID', 'alphafold_protein_id', 'esm_ProteinID'], inplace=True)

# Display the final DataFrame
print(merged_df)

#%%
# Remove unwanted FASTA sequence columns
merged_df.drop(columns=['protvec_fasta', 'alphafold_fasta', 'esm_Protein_sequence','biovec_Embedding Shape',], inplace=True)

# Check the updated columns
print(merged_df.columns)

# Rename specific columns in the DataFrame
merged_df.rename(columns={
    'biovec_FASTA Sequence': 'FASTA',
    'alphafold_aggregated_features': 'alphafold_embeddings',
    'esm_ESM_Embeddings': 'esm_embeddings'
}, inplace=True)

# Check the updated column names
print(merged_df.columns)

# Reorder columns to move 'FASTA' to the first position
merged_df = merged_df[['FASTA', 'protvec_protein_id', 'protvec_embedding', 
                       'biovec_UniProtID', 'biovec_Embeddings', 
                       'alphafold_protein_id', 'alphafold_embeddings', 
                       'esm_ProteinID', 'esm_embeddings']]

# Check the updated columns
print(merged_df.columns)

# Select the first 50 rows
first_50_rows = merged_df.head(50)

# Display or use the first 50 rows as needed
print(first_50_rows)

# Save the DataFrame to the specified directory
file_path = "/data/saiful/ePPI/all_eppi_db_embeddings50.csv"
first_50_rows.to_csv(file_path, index=False)

print(f"DataFrame successfully saved to {file_path}")

# Save the merged_df DataFrame to the specified directory
file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
merged_df.to_csv(file_path, index=False)

print(f"DataFrame successfully saved to {file_path}")