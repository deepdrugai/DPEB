#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:29:48 2025

@author: saiful
"""

# import os
# import shutil

# # Source and destination directories
# source_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/"
# destination_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings/"

# # Ensure the destination directory exists
# os.makedirs(destination_dir, exist_ok=True)

# # Get list of batch folders
# batch_folders = [folder for folder in os.listdir(source_dir) if folder.startswith("batch_") and os.path.isdir(os.path.join(source_dir, folder))]

# # Iterate through each batch folder and copy its contents
# for batch_folder in sorted(batch_folders):
#     batch_path = os.path.join(source_dir, batch_folder)
#     print(f"Copying files from {batch_folder}...")

#     # Iterate through files in the batch folder
#     for file_name in os.listdir(batch_path):
#         file_path = os.path.join(batch_path, file_name)

#         # Copy only files (skip directories if any)
#         if os.path.isfile(file_path):
#             shutil.copy(file_path, destination_dir)

#     print(f"Finished copying files from {batch_folder}.")

# print("All files have been successfully copied.")

# Get the list of files in the directory
import os
import pandas as pd

# Define the folder path
directory_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings/"

# Define the output file name
output_file = "/data/saiful/ePPI/alphafold_eppi_embeddings/obtained_alphafold_embed_prot_list.txt"

try:
    # Get the list of files in the directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith("_embedding.pkl")]

    # Save the file names to a text file
    with open(output_file, "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

    # Create a DataFrame from the file list
    df = pd.DataFrame(file_list, columns=["File Name"])
    print(df)

    print(f"List of files has been saved to {output_file}")
except FileNotFoundError:
    print(f"The directory {directory_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
    
#%%
# Check for duplicate file names
duplicate_files = df[df.duplicated(subset="File Name", keep=False)]
if not duplicate_files.empty:
    print("Duplicate file names found:")
    print(duplicate_files)
else:
    print("No duplicate file names found.")
    
    #%%
# Count files in batch_x folders
source_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/"

# Initialize a dictionary to store folder name and file count
folder_file_counts = {}

# Iterate through all batch_x folders
for folder_name in os.listdir(source_dir):
    if folder_name.startswith("batch_"):
        batch_folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(batch_folder_path):
            # Count files in the current batch folder
            file_count = len([file for file in os.listdir(batch_folder_path) if os.path.isfile(os.path.join(batch_folder_path, file))])
            folder_file_counts[folder_name] = file_count

# Print the file count for each folder
for folder, count in folder_file_counts.items():
    print(f"{folder}: {count} files")

# Save the counts to a text file
output_file_path_counts = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_file_counts.txt"
with open(output_file_path_counts, "w") as f:
    for folder, count in folder_file_counts.items():
        f.write(f"{folder}: {count} files\n")

print(f"File counts saved to {output_file_path_counts}")
#%% check any pickle file

import pickle

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZY1_embedding.pkl"

try:
    # Load the .pkl file
    with open(pkl_file_path, 'rb') as file:
        pkl_content = pickle.load(file)

    # Print the content of the .pkl file
    print(f"Content of {pkl_file_path}:")
    print(pkl_content)

except FileNotFoundError:
    print(f"The file '{pkl_file_path}' does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")


#%% load a sample pkl file as a  txt file

#%%check the shape of the embedding:

import pickle

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZY1_embedding.pkl"

try:
    # Load the .pkl file
    with open(pkl_file_path, 'rb') as file:
        pkl_content = pickle.load(file)
    
    # Extract the embedding and check its shape
    embedding = pkl_content['embedding']
    embedding_shape = embedding.shape

    print(f"Shape of the embedding: {embedding_shape}")

except FileNotFoundError:
    print(f"The file '{pkl_file_path}' does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    

#%%   load a pickle file, check its content, and save all the dictionary information (including protein_id, fasta, and embedding) into a CSV file
import os
import pickle
import numpy as np
import csv
import json

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZY1_embedding.pkl"

# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    pkl_content = pickle.load(file)

# Print the content of the .pkl file
print(f"Content of {pkl_file_path}:")
print(f"Protein ID: {pkl_content['protein_id']}")
print(f"FASTA Sequence: {pkl_content['fasta']}")
print("Embedding Matrix loaded successfully!")

# Ensure the output directory exists
output_dir = "/data/saiful/ePPI/txt_embeddings/"
os.makedirs(output_dir, exist_ok=True)

# Define the output file paths
txt_file_path = os.path.join(output_dir, f"{pkl_content['protein_id']}_embedding.txt")
csv_file_path = os.path.join(output_dir, f"{pkl_content['protein_id']}_embedding.csv")
json_file_path = os.path.join(output_dir, f"{pkl_content['protein_id']}_embedding.json")

# Convert JAX tensor (or NumPy array) to a NumPy array for safe processing
if isinstance(pkl_content['embedding'], np.ndarray):
    embedding_array = pkl_content['embedding']
else:
    embedding_array = np.array(pkl_content['embedding'])

# Save as text file
with open(txt_file_path, 'w') as txt_file:
    # Write metadata
    txt_file.write(f"Protein ID: {pkl_content['protein_id']}\n")
    txt_file.write(f"FASTA Sequence: {pkl_content['fasta']}\n")
    txt_file.write("\nEmbedding Matrix:\n")
    
    # Write embedding matrix
    for row in embedding_array:
        txt_file.write(", ".join(map(str, row)) + "\n")

print(f"Dictionary saved as text file: {txt_file_path}")

# Save as CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write metadata
    writer.writerow(["protein_id", pkl_content['protein_id']])
    writer.writerow(["fasta", pkl_content['fasta']])
    writer.writerow([])  # Blank line for separation
    
    # Write embedding matrix
    writer.writerows(embedding_array)
    
# Save as JSON file
json_content = {
    "protein_id": pkl_content['protein_id'],
    "fasta": pkl_content['fasta'],
    "embedding": embedding_array.tolist()  # Convert NumPy array to list for JSON serialization
}

with open(json_file_path, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)

print(f"Dictionary saved as CSV file: {csv_file_path}")

#%% load a saved json file to check 
import json
import numpy as np

# Path to the JSON file
json_file_path = "/data/saiful/ePPI/txt_embeddings/A0A024QZY1_embedding.json"

# Load the JSON file
with open(json_file_path, 'r') as json_file:
    json_content = json.load(json_file)

# Convert the embedding list back to a NumPy array
json_content['embedding'] = np.array(json_content['embedding'], dtype=np.float32)

# Now `json_content` is equivalent to the original `pkl_content`
print("Loaded JSON content:")
print(f"Protein ID: {json_content['protein_id']}")
print(f"FASTA Sequence: {json_content['fasta']}")
print(f"Embedding Matrix Shape: {json_content['embedding'].shape}")
#%% code to iterate through each batch_ folder, find all .pkl files, extract the shape of their embeddings, and save the results (along with the protein_id) to a text file

import os
import pickle

# Root directory containing batch folders
root_directory = "/data/saiful/ePPI/alphafold_eppi_embeddings/"
output_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/alphafold_embedding_shapes.txt"

# Open the output file for writing
with open(output_file_path, 'w') as output_file:
    # Iterate over all batch folders in the root directory
    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        if os.path.isdir(folder_path) and folder.startswith("batch_"):
            # Write the batch folder name
            output_file.write(f"Batch Folder: {folder}\n")
            output_file.write("=" * 50 + "\n")
            
            # Iterate over all .pkl files in the batch folder
            for file in os.listdir(folder_path):
                if file.endswith(".pkl"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        # Load the .pkl file
                        with open(file_path, 'rb') as pkl_file:
                            pkl_content = pickle.load(pkl_file)
                        
                        # Extract protein_id and embedding shape
                        protein_id = pkl_content['protein_id']
                        embedding_shape = pkl_content['embedding'].shape
                        
                        # Write the protein_id and embedding shape to the file
                        output_file.write(f"Protein ID: {protein_id}, Shape: {embedding_shape}\n")
                    except Exception as e:
                        # Handle any errors for specific files
                        output_file.write(f"Error processing {file}: {e}\n")
            
            # Add a blank line between batch folders
            output_file.write("\n")

print(f"Embedding shapes have been saved to {output_file_path}")


    
#%% finding the missing prot id for a batch
import os

# Define the path to the text file
input_file_path = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold/batch_34.txt"

# Read and load the proteins from the text file
with open(input_file_path, 'r') as file:
    protein_data = file.read()

# Split the content by '>' to isolate each protein and remove empty entries
proteins = [entry.strip() for entry in protein_data.split('>') if entry.strip()]

# Parse the proteins into a list of dictionaries
protein_list = []
for protein in proteins:
    lines = protein.splitlines()
    protein_id = lines[0].strip()
    fasta_sequence = ''.join(lines[1:])  # Combine all lines except the first as the sequence
    protein_list.append({'protein_id': protein_id, 'fasta': fasta_sequence})

# Display the parsed protein list (first 5 entries for brevity)
for i, protein in enumerate(protein_list[:3], 1):
    print(f"Protein {i}:")
    print(f"ID: {protein['protein_id']}")
    print(f"FASTA: {protein['fasta']}\n")


# Create a new list containing only the protein IDs
prot_id_list = [protein['protein_id'] for protein in protein_list]

prot_id_list_OG = prot_id_list[1:]

# Step 2: Extract protein IDs from the batch folder
batch_folder_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_34/"
if not os.path.exists(batch_folder_path):
    print(f"Batch folder not found: {batch_folder_path}")
else:
    # Get all files in the batch folder
    files_in_batch_folder = os.listdir(batch_folder_path)

    # Extract protein IDs from file names
    prot_id_list_from_folder = [
        filename.split("_embedding.pkl")[0] for filename in files_in_batch_folder if filename.endswith("_embedding.pkl")
    ]

    # Step 3: Find missing protein IDs
    missing_prot_ids = [prot_id for prot_id in prot_id_list_OG if prot_id not in prot_id_list_from_folder]

    # Print results
    print("Protein IDs in original list but missing in the batch folder:")
    print(missing_prot_ids)
    
    
#%%  find the missing proteins that need to run again for alphafold

# Define the file path
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/obtained_alphafold_embed_prot_list.txt"

# Initialize an empty list to store protein IDs
protein_ids_obtained = []

# Open the file and read its contents
with open(file_path, 'r') as file:
    for line in file:
        # Remove any leading/trailing whitespace (e.g., newline characters)
        line = line.strip()
        # Check if the line is not empty and does not start with "."
        if line and not line.startswith("."):
            # Split the line at "_embedding.pkl" and take the first part (protein ID)
            protein_id = line.split("_embedding.pkl")[0]
            # Append the protein ID to the list
            protein_ids_obtained.append(protein_id)

# Print the total number of protein IDs and the list
print(f"Total protein IDs: {len(protein_ids_obtained)}")
print("Protein IDs:", protein_ids_obtained)

import pandas as pd

# Define the file path
file_path = "/data/saiful/ePPI/prot_to_fasta_23k.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path, sep=",", header=None)  # Assuming the file is tab-separated

# Extract the first column (protein IDs) and convert it to a list
protein_id_total = df[0].tolist()

# Print the total number of protein IDs and the list
print(f"Total protein IDs: {len(protein_id_total)}")
print("Protein IDs:", protein_id_total)

## o find the protein IDs that are extra in protein_id_total
# Convert both lists to sets
set_protein_ids_obtained = set(protein_ids_obtained)
set_protein_id_total = set(protein_id_total)

# Find the extra protein IDs in protein_id_total
extra_proteins = set_protein_id_total - set_protein_ids_obtained

# Convert the result back to a list (if needed)
extra_proteins_list = list(extra_proteins)

# Print the number of extra proteins and the list
print(f"Number of extra proteins: {len(extra_proteins_list)}")
print("Extra proteins:", extra_proteins_list)

## save the extra_proteins_list as a text file

import os

# Define the directory and file path
output_dir = "/data/saiful/ePPI/remaining_proteins/"
output_file = os.path.join(output_dir, "protein_ids_remaining_list.txt")

# Ensure the directory exists, create it if it doesn't
os.makedirs(output_dir, exist_ok=True)

# Save the extra_proteins_list to the file
with open(output_file, 'w') as file:
    for protein_id in extra_proteins_list:
        file.write(f"{protein_id}\n")

print(f"Extra proteins list saved to: {output_file}")

## create a new DataFrame df2 that contains only the rows from df where the protein IDs in the first column match those in extra_proteins_list

# Filter the rows in df where the first column is in extra_proteins_list
df2 = df[df[0].isin(extra_proteins_list)]

# Print the new DataFrame
print("New DataFrame (df2):")
print(df2)

df2 = df2.drop(0)


# Optionally, save df2 to a new CSV file
output_csv_path = "/data/saiful/ePPI/remaining_proteins/remaining_proteins_data.csv"
df2.to_csv(output_csv_path, index=False, sep=",")
print(f"Filtered DataFrame saved to: {output_csv_path}")

## split the DataFrame df2 into 3 smaller DataFrames (each with approximately 500 rows) and save them as text files in the specified format
import numpy as np
import os

# Define the output directory
output_dir = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Split df2 into 3 smaller DataFrames
split_size = 500  # Number of rows per split
splits = np.array_split(df2, np.ceil(len(df2) / split_size))
# splits = 496, 496, 495
# Save each split as a text file in the specified format
for i, split_df in enumerate(splits):
    # Define the output file path
    file_name = f"batch_{50 + i}.txt"  # File names: batch_50.txt, batch_51.txt, batch_52.txt
    file_path = os.path.join(output_dir, file_name)
    
    # Open the file and write the data in the specified format
    with open(file_path, 'w') as file:
        for _, row in split_df.iterrows():
            protein_id = row[0]  # First column: Protein ID
            sequence = row[1]    # Second column: Protein sequence
            file.write(f">{protein_id}\n{sequence}\n")
    
    print(f"Saved: {file_path}")

print("All files saved successfully.")

#%% checking remaining protein ids fasta length of batch50, batch51, and batch52
import os

def list_protein_ids(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Extract protein IDs by removing '_embedding.pkl'
    protein_ids = [file.replace('_embedding.pkl', '') for file in files if file.endswith('_embedding.pkl')]

    return protein_ids

# original protein ids given in the list
def get_protein_ids_from_txt_file(file_path):
    # Initialize an empty list to store protein IDs
    protein_ids = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line starts with '>' indicating a protein ID
            if line.startswith('>'):
                # Extract the protein ID (remove the '>')
                protein_id = line[1:].strip()
                protein_ids.append(protein_id)

    return protein_ids

# Specify the directory
directory_50 = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_50/"
directory_51 = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_51/"
directory_52 = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_52/"

# Get the list of protein IDs obtained already
protein_ids_batch_50_obtained_embeddings = list_protein_ids(directory_50)
protein_ids_batch_51_obtained_embeddings = list_protein_ids(directory_51)
protein_ids_batch_52_obtained_embeddings = list_protein_ids(directory_52)

# Specify the file path
file_path_50 = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/batch_50.txt"
file_path_51 = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/batch_51.txt"
file_path_52 = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/batch_52.txt"

# Get the list of protein IDs
protein_ids_batch_50 = get_protein_ids_from_txt_file(file_path_50)
protein_ids_batch_51 = get_protein_ids_from_txt_file(file_path_51)
protein_ids_batch_52 = get_protein_ids_from_txt_file(file_path_52)

# Find the protein IDs absent in the obtained embeddings
absent_protein_ids_batch_50 = list(set(protein_ids_batch_50) - set(protein_ids_batch_50_obtained_embeddings))
absent_protein_ids_batch_51 = list(set(protein_ids_batch_51) - set(protein_ids_batch_51_obtained_embeddings))
absent_protein_ids_batch_52 = list(set(protein_ids_batch_52) - set(protein_ids_batch_52_obtained_embeddings))

total_absent_protein_ids = absent_protein_ids_batch_50+ absent_protein_ids_batch_51 + absent_protein_ids_batch_52

##                  ###
import pandas as pd
file_path = "/data/saiful/ePPI/prot_to_fasta_23k.csv"
df = pd.read_csv(file_path, sep=",", header=None)  
# Create the filtered DataFrame
filtered_df = df[df[0].isin(total_absent_protein_ids)]

print(filtered_df.head())
# Add a new column for the length of the FASTA sequence
filtered_df['Sequence_Length'] = filtered_df[1].apply(len)
print(filtered_df)

# Optionally, save filtered_df to a new CSV file
output_csv_path = "/data/saiful/ePPI/remaining_proteins/remaining_proteins_data_564.csv"
filtered_df.to_csv(output_csv_path, index=False, sep=",")
print(f"Filtered DataFrame saved to: {output_csv_path}")


#%%

## split the DataFrame df2 into 3 smaller DataFrames (each with approximately 500 rows) and save them as text files in the specified format
import numpy as np
import os

# Define the output directory
output_dir = "/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Split filtered_df into 3 smaller DataFrames
split_size = 1500  # Number of rows per split
splits = np.array_split(filtered_df, np.ceil(len(filtered_df) / split_size))
# splits = 496, 496, 495
# Save each split as a text file in the specified format
for i, split_df in enumerate(splits):
    # Define the output file path
    file_name = f"batch_{53 + i}.txt"  # File names: batch_50.txt, batch_51.txt, batch_52.txt
    file_path = os.path.join(output_dir, file_name)
    
    # Open the file and write the data in the specified format
    with open(file_path, 'w') as file:
        for _, row in split_df.iterrows():
            protein_id = row[0]  # First column: Protein ID
            sequence = row[1]    # Second column: Protein sequence
            file.write(f">{protein_id}\n{sequence}\n")
    
    print(f"Saved: {file_path}")

print("All files saved successfully.")

#%%
import pickle
import numpy as np
# Paths
input_pkl_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZY1_embedding.pkl"
output_pkl_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_version_independent/A0A024QZY1_embedding.pkl"


# 1. Load the original pickle file
with open(input_pkl_path, 'rb') as f:
    original_data = pickle.load(f)

# 2. Ensure all data is portable (no framework-specific tensors)
#    In your case, the embedding is already a NumPy array, which is safe.
#    We'll explicitly convert to NumPy arrays for robustness.
portable_data = {
    'protein_id': str(original_data['protein_id']),  # Ensure string type
    'fasta': str(original_data['fasta']),            # Ensure string type
    'embedding': np.array(original_data['embedding'])  # Ensure NumPy array
}

# 3. Save the portable data as a new pickle file
with open(output_pkl_path, 'wb') as f:
    pickle.dump(portable_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Portable pickle saved to: {output_pkl_path}")

#%% convert a pickle embedding to list pkl format
import pickle

# Paths
input_pkl_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZY1_embedding.pkl"
output_pkl_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_version_independent/A0A024QZY1_embeddings_plain.pkl"

# 1. Load the original pickle file
with open(input_pkl_path, 'rb') as f:
    original_data = pickle.load(f)

# 2. Convert NumPy array to nested Python lists (pure floats)
embedding_list = original_data['embedding'].tolist()  # Converts to list of floats

# 3. Save as a new pickle with pure Python types
portable_data = {
    'protein_id': original_data['protein_id'],
    'fasta': original_data['fasta'],
    'embedding': embedding_list  # Now a list, not a NumPy array
}

with open(output_pkl_path, 'wb') as f:
    pickle.dump(portable_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {output_pkl_path}. No dependencies needed to load!")

#%% check any pickle file

import pickle

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/A0A024QZS4_embedding.pkl"
# pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_version_independent/A0A024QZS4_embedding.pkl"
# pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_version_independent/A0A024QZY1_embeddings_plain.pkl"
try:
    # Load the .pkl file
    with open(pkl_file_path, 'rb') as file:
        pkl_content = pickle.load(file)

    # Print the content of the .pkl file
    print(f"Content of {pkl_file_path}:")
    print(pkl_content)

except FileNotFoundError:
    print(f"The file '{pkl_file_path}' does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    
#%% batch process all .pkl files in the input directory and save them to the version-independent format

import os
import pickle
import numpy as np

# Configuration
input_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings/"
output_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/"

# input_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_1/"
# output_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_version_independent/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all pkl files in input directory
for filename in os.listdir(input_dir):
    if filename.endswith("_embedding.pkl"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load original data
            with open(input_path, 'rb') as f:
                original_data = pickle.load(f)
            
            # Create portable version
            portable_data = {
                'protein_id': str(original_data['protein_id']),
                'fasta': str(original_data['fasta']),
                'embedding': np.array(original_data['embedding'])
            }
            
            # Save converted file
            with open(output_path, 'wb') as f:
                pickle.dump(portable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Converted: {filename}")
            
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")
            continue

print("\nBatch conversion complete!")

#%% convert all jax pickle files t version independent numpy pickle files
import os
import pickle
import numpy as np

# Configuration
input_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings/"
output_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize counters
success_count = 0
fail_count = 0

# Process all pkl files in input directory
for filename in os.listdir(input_dir):
    if filename.endswith("_embedding.pkl"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load original data
            with open(input_path, 'rb') as f:
                original_data = pickle.load(f)
            
            # Create portable version
            portable_data = {
                'protein_id': str(original_data['protein_id']),
                'fasta': str(original_data['fasta']),
                'embedding': np.array(original_data['embedding'])
            }
            
            # Save converted file
            with open(output_path, 'wb') as f:
                pickle.dump(portable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            success_count += 1
            print(f"Converted: {filename} (Success #{success_count})")
            
        except Exception as e:
            fail_count += 1
            print(f"Failed to process {filename}: {str(e)} (Failure #{fail_count})")
            continue

# Print summary
print("\nBatch conversion complete!")
print(f"Total files processed: {success_count + fail_count}")
print(f"Successfully converted: {success_count}")
print(f"Failed conversions: {fail_count}")

#%% copy batch50,51 and 52 pickle files to All_ePPI_Alphafold2_Embeddings folder
import os
import shutil

# Configuration
source_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_52/"
dest_dir = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings/"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Initialize counters
copied_files = 0
skipped_files = 0

# Copy all files
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    
    # Skip directories, only process files
    if os.path.isfile(source_path):
        try:
            shutil.copy2(source_path, dest_path)  # copy2 preserves metadata
            copied_files += 1
            print(f"Copied: {filename}")
        except Exception as e:
            skipped_files += 1
            print(f"Failed to copy {filename}: {str(e)}")
    else:
        skipped_files += 1
        print(f"Skipped directory: {filename}")

# Print summary
print("\nCopy operation complete!")
print(f"Total files attempted: {copied_files + skipped_files}")
print(f"Successfully copied: {copied_files}")
print(f"Skipped/not copied: {skipped_files}")

#%% single_sequence vs custom

import pickle

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_x/A0A024QZJ7_embedding_custom.pkl"
# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    custom_pkl_content = pickle.load(file)

# Print the content of the .pkl file
print(f"Content of {pkl_file_path}:")
print(custom_pkl_content)

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_x/A0A024QZJ7_embedding_ss.pkl"
# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    ss_pkl_content = pickle.load(file)

# Print the content of the .pkl file
print(f"Content of {pkl_file_path}:")
print(ss_pkl_content)

#%% single_sequence vs custom2 A0A024R0L5
import pickle

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_x/A0A024R0L5_embedding_custom.pkl"
# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    custom_pkl_content = pickle.load(file)

# Print the content of the .pkl file
print(f"Content of {pkl_file_path}:")
print(custom_pkl_content)

# Path to the specific .pkl file
pkl_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/batch_x/A0A024R0L5_embedding_ss.pkl"
# Load the .pkl file
with open(pkl_file_path, 'rb') as file:
    ss_pkl_content = pickle.load(file)

# Print the content of the .pkl file
print(f"Content of {pkl_file_path}:")
print(ss_pkl_content)

#%%
import pickle

# Path to the .pkl file
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/All_ePPI_Alphafold2_Embeddings_np/A0A024QZN9_embedding.pkl"

# Load the .pkl file
with open(file_path, 'rb') as file:
    embedding_data = pickle.load(file)

# Print or inspect the loaded data
print(embedding_data)
