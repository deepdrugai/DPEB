This directory contains scripts to evaluate protein embeddings through clustering, followed by downstream classification tasks.

#### Prerequisites
Before running the scripts, ensure you have the environment set up as described in the root directory's 'DPEB.yml' file.

'protein_families23k.csv': The dataset containing protein sequences and family labels.

#### Execution Workflow
To achieve the full pipeline (supervised training followed by unsupervised evaluation and classification),  run the scripts in the following order based on your chosen embedding model:

#### (ESM)
1. Run the supervised training: python 'supervised_clustering - esm.py'
2. Run the unsupervised evaluation and classification: python 'unsupervised_clustering_and_downstream_classification_ESM.py'
