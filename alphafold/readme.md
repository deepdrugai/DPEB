

##  Extracting AlphaFold Embeddings

This codebase is adapted from [xinformatics/alphafold](https://github.com/xinformatics/alphafold), with additional modifications to extract embeddings for custom protein datasets.

Follow the instructions below to extract embeddings for a batch of protein sequences.



###  Step 1: Create Output Directory

Create the output directory where the embeddings and console logs will be saved:

```bash
mkdir -p /path/to/output/alphafold_embeddings/batch_X
mkdir -p /path/to/output/alphafold_embeddings/console_output
```

Replace `/path/to/output/` with your actual desired path, and `batch_X` with your batch identifier (e.g., `batch_40`).

---

###  Step 2: Modify `run_alphafold.sh`

Edit the following file:

```
alphafold/run_alphafold.sh
```

Update the paths in the script:

```bash
# Redirect stdout and stderr to a log file for the specific batch
exec > /path/to/output/alphafold_embeddings/console_output/console_output_batch_X.txt 2>&1

# Set the input FASTA file containing protein sequences for the current batch
SEQUENCES_FILE="/path/to/fasta_batches/batch_X.txt"
```

---

###  Step 3: Modify Output Path in Embedding Script

Edit the following script:

```
alphafold/Representations_AlphaFold2_v3.23.py
```

Update the line that saves the protein embeddings:

```python
with open(f'/path/to/output/alphafold_embeddings/batch_X/{protein_id}_embedding.pkl', 'wb') as handle:
```

This ensures the embeddings are saved in the correct batch-specific folder.

---

###  Step 4: Run the Embedding Extraction Pipeline

Execute the modified AlphaFold pipeline:

```bash
bash alphafold/run_alphafold.sh
```

This will generate `.pkl` embedding files for each protein in the specified batch.

---

###  Notes

- Ensure AlphaFold is installed and all dependencies are correctly configured.
- Input FASTA files must be organized batch-wise.
- The output `.pkl` files contain protein-level embeddings and can be used for downstream tasks such as protein classification, protein-protein interaction prediction, etc.
```

