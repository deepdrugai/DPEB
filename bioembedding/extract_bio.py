from bio_embeddings.embed import ProtTransBertBFDEmbedder
import pandas as pd
import pickle
import time
import os
import lmdb
import hashlib
import numpy as np
from tqdm import tqdm

# start_time = time.time()
# Application Execution
df_human_prot_fasta = pd.read_csv("pro_id_seq_human.csv")
print(df_human_prot_fasta.columns)    

human_prot_red_embeddings = []
output_dir = "/home/magesh/protein_embeddings_npy"
# #sequences = sequences[:20]
embedder = ProtTransBertBFDEmbedder()
batch_size = 1024
start, end = 0, 2 # 10000 Embeddings
# # start, end = 10000, 16772 # 6772
# #start, end = 16772, 23019 # 6247
# for i in range(start, end, batch):
def iterate_batches(df_human_prot_fasta, batch_size):
    for start in range(0, len(df_human_prot_fasta), batch_size):
        yield df_human_prot_fasta[start:start + batch_size]
dict_list = []

# P35225.npy - > read : (seq_len, 1024) shape

for i, df_batch in tqdm(enumerate(iterate_batches(df_human_prot_fasta, batch_size))):
    protein_ids = df_batch["ProteinID"].tolist()
    sequences = df_batch["Protein_sequence"].tolist()

    # Generate per-residue embeddings
    embeddings = embedder.embed_many(sequences)
    # print(f"Batch {i} processed with shape of the {embeddings.shape}.")

    # Save each as a dict in .npy format
    for prot_id, seq, emb in zip(protein_ids, sequences, embeddings):
        print(f"Protein: {prot_id} processed with shape of the {emb.shape}.")
        out_dict = {
            "protein_id": prot_id,
            "fasta": seq,
            "embedding": emb  # shape: (L, 1280)
        }
        file_path = os.path.join(output_dir, f"{prot_id}_embedding.npy")
        np.save(file_path, out_dict)   

    #Commented everything below for now to run each protein sequence separately
    # seq = sequences[i:i+batch]
    # uniprot = uniprot_ids[i:i+batch]
    # print(df_batch['Protein_sequence'])
    # human_prot_embeddings = embedder.embed_many([s for s in df_batch['Protein_sequence']])
    # human_list = list(human_prot_embeddings)
    # print(len(human_list))
    # # human_prot_embeddings = list(human_prot_embeddings)
    # # reduced_embeddings_f = [ProtTransBertBFDEmbedder.reduce_per_protein(e) for e in human_prot_embeddings]
    # # dict_list.append(dict(zip(df_batch['Protein_sequence'], reduced_embeddings_f)))
    # # print(list(reduced_embeddings_f))
    # for per_amino_acid in human_list:
    #     print(i, per_amino_acid.shape)

    # uni_dict = dict(zip(df_batch['ProteinID'], reduced_embeddings_f))
    # human_prot_red_embeddings.append(reduced_embeddings_f)
    # print("Finished Batch Embedding {}".format(i))
    # flatten_list = [x for xs in human_prot_red_embeddings for x in xs]


    # name = "human_prot_bio_embeddings_final_" + str(start) + "_" + str(end) + "_.pkl"
    # with open(name, 'wb') as f:
    #   pickle.dump(flatten_list, f)

# env = lmdb.open(f'bio_emb_protein_full_list.lmdb', map_size=10*1024*1024*1024)

# def hash_fasta_id(fasta):
#     # Generate a SHA-256 hash for the UniProt ID.
#     return hashlib.sha256(fasta).hexdigest()

# with env.begin(write=True) as txn:
#     for i in dict_list:
#         for (seq, emb) in i.items():
#             # print(seq)
#             uniprot_id = df_human_prot_fasta.loc[df_human_prot_fasta['Protein_sequence'] == seq, "ProteinID"].values[0]
#             # print(uniprot_id)
#             uniprot_id = f"{uniprot_id}".encode('utf-8')
#             fasta_seq = f"{seq}".encode('utf-8')
#             value = {
#                 'uniprot_id': uniprot_id,
#                 'fasta_seq': seq,
#                 'embeddings': emb
#             }
#             # txn.put(f"{hash_fasta_id(uniprot_id)}_embeddings".encode(), pickle.dumps(value, protocol=-1))
#             txn.put(f"{hash_fasta_id(fasta_seq)}_embeddings".encode(), pickle.dumps(value, protocol=-1))
# env.close()
print("Data successfully stored!")