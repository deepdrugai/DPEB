
## Protein Bio-Embeddings Generator

This script generates **per-residue protein embeddings** using the **ProtTransBert-BFD** model from the `bio_embeddings` library.

Each protein sequence is embedded and saved as an individual `.npy` file.

---

#### Input File

The script expects a CSV file named: `pro_id_seq_human.csv`


#### Required Columns

| Column | Description |
|------|------------|
| ProteinID | Unique protein identifier (e.g., UniProt ID) |
| Protein_sequence | Amino-acid sequence |

Example:
```csv
ProteinID,Protein_sequence
P35225,MADSASESDTDGAGGNSSSSAAMQSS...

####Environment Setup
Create and activate a Conda environment:

```bash
conda create -n bioemb python=3.8 -y
conda activate bioemb

```

Install dependencies:

```bash

pip install bio-embeddings pandas numpy tqdm lmdb

```
A CUDA-enabled GPU is strongly recommended for reasonable runtime.

#### Configuration
Inside `extract_bio.py`, set the output directory:

```python
output_dir = "/home/magesh/protein_embeddings_npy"

```

Adjust batch size if GPU memory is limited:
```python
batch_size = 1024  # reduce if CUDA OOM occurs

```

#### Run the Script
From the repository directory:
```bash

python extract_bio.py

```

#### Output
Each protein produces one `.npy` file:


```bash

P35225_embedding.npy

```

Each file contains a Python dictionary:

```python

{
  "protein_id": str,
  "fasta": str,
  "embedding": np.ndarray  # shape: (sequence_length, 1024)
}

```
#### Load an Embedding

```python

{

import numpy as np

data = np.load("P35225_embedding.npy", allow_pickle=True).item()
embedding = data["embedding"]

}

```



##### Notes
- Embeddings are per-residue (not pooled)
- Embedding dimension: 1024
- Suitable for PPI, GNNs, clustering, and other protein ML tasks






