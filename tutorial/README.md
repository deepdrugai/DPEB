
# Tutorial: Aggregating and Clustering AlphaFold2 Embeddings from DPEB

This tutorial demonstrates how to load `.npy` embedding files from the [DeepDrug Protein Embeddings Bank (DPEB)](https://github.com/deepdrugai/DPEB), perform **mean pooling**, and apply **clustering** and **t-SNE visualization** to analyze protein families.

---

## Objective



This script will perform the following actions:

- Load **AlphaFold2 `.npy` embedding files** stored in the dataset
- Apply **mean pooling** across residue-level embeddings to produce fixed-length vectors
- Aggregate per-residue embeddings into fixed-length vectors using **mean pooling**
- Merge these embeddings with protein family labels from `protein_families23k.csv`
- Save the aggregated results to `eppi_alphafold_aggregated_embeddings.csv`
- Generate **t-SNE plots** and save them as:
  - `raw_embeddings_tsne_Alphafold.png`
  - `raw_embeddings_kmeans_tsne_Alphafold.png`
- Print **clustering evaluation metrics** (Accuracy, Precision, Recall, F1-Score)
---

##  Input Files

- AlphaFold2 embeddings: `.npy` files inside a `.rar` archive
- Protein family annotations: `protein_families23k.csv`
- Output aggregated embeddings: `eppi_alphafold_aggregated_embeddings.csv`

---
##  Files in This Folder

- **`tutorial_clustering.py`**: Main script that performs loading, aggregation, clustering, and visualization.
- **`protein_families23k.csv`**: CSV file mapping each protein to its family (used for supervised evaluation).
- **Output**:
  - `eppi_alphafold_aggregated_embeddings.csv` (generated), we have also provided this inside each embedding folder
  - `raw_embeddings_tsne_Alphafold.png` (generated)
  - `raw_embeddings_kmeans_tsne_Alphafold.png` (generated)

##  Required Libraries

- `numpy`, `pandas`
- `matplotlib`, `scikit-learn`
- `torch`, `TSNE`, `KMeans`
- `ast`, `LabelEncoder`

Activate your conda environment and install any missing dependencies.

```bash
conda activate DPEB
pip install scikit-learn matplotlib pandas torch
```
## Run the Tutorial Script

You can execute the tutorial by running the Python script inside the `tutorial/` folder:

```bash
python tutorial_clustering.py
```

