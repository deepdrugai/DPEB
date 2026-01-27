

Please ensure you have the conda  environment set up as defined in the root env.yml file.

- main_concat.py: Run this script for training and evaluating the PPI model using concatenated embeddings.

- model.py: Defines the neural network architecture used for the interaction prediction.

- concat_embeddings_robustness/: Contains results or scripts related to the robustness testing of the concatenation method.

- console_outputs/: Logs and terminal outputs from previous training runs for reference.


## Protein–Protein Interaction Prediction (Concatenated Embeddings)
This directory contains the implementation for predicting Protein-Protein Interactions (PPI) by concatenating 4 different types of protein embeddings.


### Environment Setup
Please ensure that the conda environment is set up using the `env.yml` file located at the root of the repository.

### Directory Structure

- **main_concat.py**  
  Main script for training and evaluating the PPI prediction model using concatenated embeddings.

- **model.py**  
  Defines the neural network architecture used for interaction prediction.

- **concat_embeddings_robustness/**  
  Contains scripts and/or results related to robustness analysis of the embedding concatenation approach.

- **console_outputs/**  
  Stores logs and terminal outputs from previous training and evaluation runs for reference.

