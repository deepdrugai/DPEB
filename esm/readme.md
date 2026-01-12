# ESM-2 Protein Embedding Extraction Pipeline

This repository provides a simple, reproducible pipeline to generate **ESM-2 embeddings** for protein sequences using **ESM (Evolutionary Scale Modeling)** models.

The pipeline is designed for researchers who already have protein sequences in **FASTA or CSV format** and want to extract **per-residue or full-sequence embeddings** for downstream tasks such as:
- Protein–protein interaction (PPI)
- Function prediction
- Clustering and visualization
- Graph neural networks
- Machine learning on protein embeddings

---



### File descriptions

- **`esm2_input.fasta`**  
  Example FASTA input file containing protein IDs and sequences.

- **`get_esm_embeddings.py`**  
  Main script for generating ESM-2 embeddings from your own protein sequences.  
  This is the recommended and maintained pipeline.

- **`esmfold2_embeddings.py`**  
  Legacy / experimental notebook-style script.  
  Not recommended for reuse or adaptation.

---

## Requirements

### 1. Conda environment (recommended)

Create and activate an environment with PyTorch and ESM installed.

```bash
conda create -n esm_env python=3.8 -y
conda activate esm_env
```
Install PyTorch (adjust CUDA version if needed):
```bash
pip install torch torchvision torchaudio

```
Install ESM:
```bash
pip install fair-esm

```

#### Input Format
FASTA file format
Your input FASTA file must look like this:

```fasta
>A0A024QZJ7
MADSASESDTDGAGGNSSSSAAMQSSCSSTSGGGGGGGGGGGGGKSGGIVISPFRLEELTNRL...
>A0A024QZN9
MATHGQTCARRNWCNELRLPALKQHSIGRGLESHITMCIPPSYADLGKAARDIFNKGFGFGL...

```
Requirements:
- Each protein must have a unique ID
- Sequences must be single-line, no spaces
- Standard amino acid alphabet only

#### Step 1: Generate ESM `.pt` embedding files
Use the official esm-extract command.

```bash

esm-extract esm2_t33_650M_UR50D \
  esm2_input.fasta \
  esm_pt_files/ \
  --repr_layers 33 \
  --include mean per_tok

```
Explanation:

- `esm2_t33_650M_UR50D` → ESM-2 model (33 layers, 650M parameters)

- `esm2_input.fasta` → Your input sequences

- `esm_pt_files/` → Output directory for .pt files

- `--repr_layers 33` → Extract embeddings from the final layer

- `mean` → Full-sequence embedding

- `per_tok` → Per-residue embeddings

After this step, you should see files like:
```bash
esm_pt_files/
├── A0A024QZJ7.pt
├── A0A024QZN9.pt
└── ...
```
#### Step 2: Convert `.pt` files into NumPy embeddings
Edit paths inside `get_esm_embeddings.py` if needed:

```python

csv_path = "/path/to/prot_to_fasta.csv"
fasta_path = "./esm2_input.fasta"
esm2_embedding_dir = "/path/to/esm_pt_files"
save_dir = "/path/to/esm2_dict_embeddings"
EMB_LAYER = 33

```
Then run:
```bash

python get_esm_embeddings.py

```
#### Output Format
For each protein, the script saves a .npy file containing a dictionary:
```python

{
  "protein_id": "A0A024QZJ7",
  "fasta": "MADSASESDTDGAGG...",
  "embedding": np.ndarray  # shape: (L, 1280)
}

```
Where:

- L = protein sequence length
- 1280 = ESM-2 embedding dimension

  Example output directory:
```bash

esm2_dict_embeddings/
├── A0A024QZJ7_esm2.npy
├── A0A024QZN9_esm2.npy
└── ...


```

#### Loading Embeddings in Python

```python
import numpy as np

data = np.load("A0A024QZJ7_esm2.npy", allow_pickle=True).item()

print(data["protein_id"])
print(len(data["fasta"]))
print(data["embedding"].shape)


```
