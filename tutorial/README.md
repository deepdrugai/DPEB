
# Tutorial: Aggregating and Clustering AlphaFold2 Embeddings from DPEB

This tutorial demonstrates how to load `.npy` embedding files from the [DeepDrug Protein Embeddings Bank (DPEB)](https://github.com/deepdrugai/DPEB), perform **mean pooling**, and apply **clustering** and **t-SNE visualization** to analyze protein families.

---

## Objective

We show how to:
- Load AlphaFold2 `.npy` embedding files
- Aggregate per-residue embeddings into fixed-length vectors using **mean pooling**
- Merge the embeddings with protein family metadata
- Apply **K-Means clustering**
- Visualize using **t-SNE**
- Evaluate clustering performance (Accuracy, Precision, Recall, F1-Score)

---

##  Input Files

- AlphaFold2 embeddings: `.npy` files inside a `.rar` archive
- Protein family annotations: `protein_families23k.csv`
- Output aggregated embeddings: `eppi_alphafold_aggregated_embeddings.csv`

---

##  Required Libraries

- `numpy`, `pandas`
- `matplotlib`, `scikit-learn`
- `torch`, `TSNE`, `KMeans`
- `ast`, `LabelEncoder`

Activate your conda environment and install any missing dependencies.

```bash
conda activate DPEB
pip install scikit-learn matplotlib pandas torch

