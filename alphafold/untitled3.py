#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:19:35 2025

@author: saiful
"""
# check any pickle file
import pickle

# Specify the path to your file
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/A0A024QZJ7_embedding.pkl"

# Load the Pickle file
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Print the keys and their values
if isinstance(data, dict):
    print("Keys and Values in the Pickle file:")
    for key, value in data.items():
        print(f"Key: {key}")
        print(f"Value: {value}\n")
else:
    print("The Pickle file doesn't contain a dictionary.")
#%%  convert your Pickle files (.pkl) to NumPy files (.npy), 

import pickle
import numpy as np
import os

# Paths
input_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/A0A024QZJ7_embedding.pkl"
output_folder = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np_v1.3/"
output_file_name = "A0A024QZJ7_embedding.npy"  # Naming the output file

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load Pickle file
with open(input_file_path, "rb") as file:
    data = pickle.load(file)

# Check if the 'embedding' key exists
if "embedding" in data:
    embedding_data = np.array(data["embedding"])  # Convert to NumPy array
    output_file_path = os.path.join(output_folder, output_file_name)
    
    # Save as NumPy file
    np.save(output_file_path, embedding_data)
    print(f"Embedding data saved successfully to {output_file_path}")
else:
    print("The 'embedding' key was not found in the Pickle file.")
    
#%% load the .npy file to check if it was properly converted
import numpy as np

# Specify the path to the .npy file
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np1.3/A0A024QZJ7_embedding.npy"

# Load the NumPy file
data = np.load(file_path)

# Check the data
print("Shape of the array:", data.shape)
print("Data type:", data.dtype)
print("First few values:")
print(data[:5])  # Print the first 5 rows (or elements)

# Confirm successful loading
print("NumPy file loaded successfully!")

#%% iterates through all .pkl files in the specified folder, converts their embedding data to .npy format, and saves the converted files into the target directory
import pickle
import numpy as np
import os

# Define paths
input_folder = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/"
output_folder = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np_v1.3/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all .pkl files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".pkl"):  # Check if the file is a Pickle file
        input_file_path = os.path.join(input_folder, file_name)
        
        # Load the Pickle file
        with open(input_file_path, "rb") as file:
            data = pickle.load(file)
        
        # Save the entire dictionary to .npy format
        output_file_name = file_name.replace(".pkl", ".npy")
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # Save dictionary as .npy file
        np.save(output_file_path, data)
        print(f"Converted and saved: {output_file_path}")
  
#%%
#  load the .npy files to ensure they contain the expected dictionary structure:
import numpy as np

# Example file to load
# file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np_v1.3/A0A024QZJ7_embedding.npy"
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np_v1.3/A0A024R0R4_embedding.npy"
# Load the NumPy file
data = np.load(file_path, allow_pickle=True)  # Use `allow_pickle=True` since it's a dictionary

# Check the data
print("Keys and Values in the NumPy file:")
if isinstance(data.item(), dict):  # `data.item()` retrieves the original dictionary
    for key, value in data.item().items():
        print(f"Key: {key}")
        print(f"Value: {value}\n")
else:
    print("The loaded data is not a dictionary.")