#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:17:33 2025

@author: saiful
"""


import lmdb
import pickle
import numpy as np
import pandas as pd
import json  # To serialize embedding lists as JSON strings

# Define the LMDB path
lmdb_path = "/home/saiful/ePPI_dgl/bioembedding/bio_emb_protein_full_list.lmdb/bio_emb_protein_full_list.lmdb"

# Open the LMDB database in read-only mode
env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=1)

entries = []
count = 0

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        data = pickle.loads(value)  # Deserialize the entry (expected to be a dictionary)
        
        # Extract the fields
        uniprot_id = data.get('uniprot_id', b'').decode('utf-8')
        fasta_seq = data.get('fasta_seq', '')
        embeddings = data.get('embeddings', None)
        
        # Convert embeddings to list if it's a numpy array
        if isinstance(embeddings, np.ndarray):
            emb_list = embeddings.tolist()
        else:
            emb_list = None
        
        # Alternatively, you can store the embeddings as a JSON string
        emb_json = json.dumps(emb_list) if emb_list is not None else None
        
        # Append the record as a dictionary
        entries.append({
            'UniProt ID': uniprot_id,
            'FASTA Sequence': fasta_seq,
            'Embedding Shape': str(embeddings.shape) if emb_list is not None else None,
            'Embeddings': emb_json  # Save the embeddings as a JSON string
        })
        
        # count += 1
        # if count >= 5:
        #     break

env.close()

# Create a DataFrame and save to CSV
df = pd.DataFrame(entries)
csv_file = "bio_embeddings_ePPI.csv"
df.to_csv(csv_file, index=False)

print(f"Saved first 5 entries with embeddings to {csv_file}")
