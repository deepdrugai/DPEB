# ProtVec Embedding Generator

This repository provides a simple and reproducible way to generate **ProtVec embeddings** for protein sequences using a **pretrained Word2Vec model**.  
Each protein is converted into **raw k-mer–level embeddings (3-mers)** and saved as a Python dictionary in `.npy` format.

This script is suitable for downstream tasks such as **protein–protein interaction (PPI) prediction, graph neural networks (GNNs), Transformers, or other deep learning pipelines**.

---

## What this script does

For each protein sequence, the script:

- Splits the sequence into overlapping **3-mers**
- Uses **three reading-frame offsets** (0, 1, 2)
- Looks up embeddings from a pretrained ProtVec model
- Stores **raw k-mer embeddings** (shape: `N × 100`)
- Saves **one `.npy` file per protein**

Each saved file contains a dictionary with the following keys:

```python
{
  "protein_id": "<protein_id>",
  "fasta": "<protein_sequence>",
  "embedding": numpy.ndarray  # shape: (num_kmers, 100)
}


Requirements
1. Create and activate a conda environment (recommended)
conda create -n ePPI4 python=3.8
conda activate ePPI4

2. Install required Python packages
pip install numpy pandas gensim

Pretrained ProtVec model

You need a pretrained ProtVec Word2Vec model (trained on protein 3-mers, e.g., SwissProt).

In the script, the model is loaded as:

model = Word2Vec.load(
    "/home/saiful/ePPI_dgl/protvec/swissprot-reviewed-protvec.model"
)


You may change this path to wherever your ProtVec model is stored.

Important assumptions about the model:

Tokenization uses 3-mers

Embedding dimension is 100

Model is compatible with gensim.models.Word2Vec

Input format

The script expects a CSV file with two columns:

Column name	Description
protein_id	Unique protein identifier
fasta	Protein sequence (amino acids)
Example CSV (prot_to_fasta.csv)
protein_id,fasta
P12345,MKVLYNLKDGKVT...
Q9XYZ1,MSDTQLERK...

How to run the script
1. Clone the repository
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>

2. Edit paths inside the script

Open get_protvec_embeddings1.2.py and update the following paths:

# Path to ProtVec model
model = Word2Vec.load("PATH_TO_PROTVEC_MODEL")

# Path to input CSV
file_path = "PATH_TO_PROT_TO_FASTA_CSV"

# Output directory
output_dir = "PATH_TO_OUTPUT_DIRECTORY"

3. Run the script
python get_protvec_embeddings1.2.py

Output

One .npy file is generated per protein

File naming format:

<protein_id>_protvec.npy

Example:
A0A024R0L9_protvec.npy


Each file stores:

Protein ID

FASTA sequence

Raw ProtVec k-mer embeddings (N × 100)

Verifying the output

You can load and inspect an embedding file using:

import numpy as np

data = np.load("A0A024R0L9_protvec.npy", allow_pickle=True).item()
print("Protein ID:", data["protein_id"])
print("FASTA length:", len(data["fasta"]))
print("Embedding shape:", data["embedding"].shape)


The script also includes helper code to randomly sample embedding files and print their shapes for quick sanity checks.

Notes

This script does NOT pool or average embeddings; it preserves raw k-mer–level information

Suitable for:

GNN-based protein modeling

Transformer-based sequence models

Custom pooling or attention mechanisms

Scales to large datasets (e.g., 20K+ proteins)