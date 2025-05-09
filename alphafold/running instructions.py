#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:28:08 2024

@author: saiful
"""

running instructions:

1.   create a folder manually "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_40/"


2.  modify the file : /home/saiful/ePPI_dgl/alphafold/run_alphafold.sh :
    - line exec > /data/saiful/ePPI/alphafold_eppi_embeddings/console_output/console_output_batch_46.txt 2>&1
    - line    SEQUENCES_FILE="/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold/batch_46.txt"
     



3. modify the line  at /home/saiful/ePPI_dgl/alphafold/Representations_AlphaFold2_v3.23.py 
- line     with open(f'/data/saiful/ePPI/alphafold_eppi_embeddings/batch_46/{protein_id}_embedding.pkl', 'wb') as handle: