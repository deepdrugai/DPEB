
### Protein–Protein Interaction Prediction (Concatenated Embeddings)

This directory contains the implementation for Protein–Protein Interaction (PPI) prediction using concatenation of different combinations of four protein embedding types. Multiple embedding subsets are combined to study their impact on interaction prediction performance.

#### Environment Setup
Please ensure that the conda environment is set up using the `env.yml` file located at the root of the repository.

#### Directory Structure

- **main_concat.py**  
  Main script for training and evaluating the PPI prediction model using concatenated embeddings.

- **model.py**  
  Defines the neural network architecture used for interaction prediction.

- **concat_embeddings_robustness/**  
  Contains results obtained related to robustness analysis of the embedding concatenation approach.

- **console_outputs/**  
  Stores logs and terminal outputs from previous training and evaluation runs for reference.

