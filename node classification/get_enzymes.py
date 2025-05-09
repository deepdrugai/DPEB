#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:49:48 2025

@author: saiful
"""


import requests

# List of protein IDs to check
protein_ids = [
    "P46926", "P43631", "D6RGX4.1", "E5RG02.1", "H3BUK9",
    "Q96S59", "Q9NTJ3", "O60828", "P59998", "O15145"
]

def check_enzyme_status(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Check for EC number in the protein's annotations
        if 'EC number' in data:
            ec_numbers = ', '.join(data['EC number'])
            return f"{protein_id} is an enzyme with EC number(s): {ec_numbers}"
        else:
            return f"{protein_id} is NOT classified as an enzyme"
    else:
        return f"Failed to retrieve data for {protein_id}"

# Check each protein ID
for pid in protein_ids:
    print(check_enzyme_status(pid))
    
#%%
import pandas as pd
import requests

# Load your DataFrame
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df_subset = pd.read_csv(file_path)

# Select the first 200 rows
# df_subset = df.head(2000)

# Function to check if a protein is an enzyme
def is_enzyme(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Check for EC number in the protein's annotations
            if 'ecNumbers' in data:
                return 'Yes'
            else:
                return 'No'
        elif response.status_code == 404:
            return 'Not Found'
        else:
            print(f"Unexpected status code {response.status_code} for {protein_id}")
            return 'Error'
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {protein_id}: {e}")
        return 'Error'

# Apply the function to the 'protein_id' column of the subset DataFrame
df_subset['Enzyme'] = df_subset['protein_id'].apply(is_enzyme)

# Update the original DataFrame with the 'Enzyme' column for the first 200 rows
df.loc[df_subset.index, 'Enzyme'] = df_subset['Enzyme']

# Display the updated DataFrame
print(df.head(205))

    #%% working 
import requests

def check_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        if "EC=" in response.text:
            return True
    return False

protein_ids = ["P00351", "P04041", "P19367", "P00338", "P00798", 
               "P00441", "P09865", "P22303", "P11413", "P00423", 
               "P46926", "P43631", "D6RGX4.1", "E5RG02.1", "H3BUK9", 
               "Q96S59", "Q9NTJ3", "O60828", "P59998", "O15145"]

results = {}
for protein_id in protein_ids:
    is_enzyme = check_uniprot(protein_id)
    results[protein_id] = "Enzyme" if is_enzyme else "Not an enzyme"

for protein_id, status in results.items():
    print(f"{protein_id}: {status}")
    
#%%
import requests
import pandas as pd
import requests

# Load your DataFrame
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df = pd.read_csv(file_path) 
# Select the first 200 rows
df_subset = df.head(50)

def check_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        if "EC=" in response.text:
            return True
    return False

# Apply the function to the 'protein_id' column of the subset DataFrame
df_subset['Enzyme'] = df_subset['protein_id'].apply(check_uniprot)

# Update the original DataFrame with the 'Enzyme' column for the first 200 rows
df.loc[df_subset.index, 'Enzyme'] = df_subset['Enzyme']

# Display the updated DataFrame
print(df.head(205))


#%%
import requests

def check_merops(protein_id):
    url = f"https://www.ebi.ac.uk/merops/cgi-bin/pepsum?id={protein_id}"
    response = requests.get(url)
    if "No peptidase found" in response.text:
        return False
    return True

# Example usage
protein_id = "S01.001"
is_enzyme = check_merops(protein_id)
print(f"MEROPS: Is {protein_id} a peptidase? {is_enzyme}")
#%%


import requests
import pandas as pd

# Load your DataFrame
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df = pd.read_csv(file_path)

# Select the first 50 rows
# df_subset = df.head(50)

def check_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.txt"
    response = requests.get(url)
    
    if response.status_code == 404:
        return "Not Found"
    elif response.status_code == 200:
        return True if "EC=" in response.text else False
    else:
        return "Error"

# Apply the function to the 'protein_id' column of the subset DataFrame
df_subset['Enzyme'] = df_subset['protein_id'].apply(check_uniprot)

# Update the original DataFrame with the 'Enzyme' column for the first 50 rows
df.loc[df_subset.index, 'Enzyme'] = df_subset['Enzyme']

# Display the updated DataFrame
# Save the updated DataFrame as a CSV file
output_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_updated_embeddings.csv"
df.to_csv(output_file_path, index=False)

# Display confirmation message
print(f"Updated dataset saved as: {output_file_path}")
print("execution finished.. ")

#%% Prints a counter every 500 proteins processed, Displays the running protein ID every 500 iterations.

import requests
import pandas as pd
import time
import os

# Load your DataFrame
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df = pd.read_csv(file_path)
# df = df.head(300)


def check_uniprot(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.txt"
    response = requests.get(url)

    if response.status_code == 404:
        return "Not Found"
    elif response.status_code == 200:
        return True if "EC=" in response.text else False
    else:
        return "Error"

# Initialize counter and timer
start_time = time.time()
batch_size = 100

# Apply the function to the 'protein_id' column with a counter
enzyme_results = []
for idx, protein_id in enumerate(df['protein_id'], start=1):
    result = check_uniprot(protein_id)
    enzyme_results.append(result)

    # Print status every 500 proteins
    if idx % batch_size == 0:
        elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
        print(f"Processed {idx} proteins. Last Protein ID: {protein_id} | Time taken: {elapsed_time:.2f} minutes")
        start_time = time.time()  # Reset timer

# Assign the results back to the DataFrame
df['Enzyme'] = enzyme_results

# Define the parent directory to save the file
parent_dir = os.path.dirname(os.path.dirname(file_path))
output_file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_enzym_info_all.csv"

# output_file_path = os.path.join(parent_dir, "eppi_alphafold_updated_embeddings.csv")
df = df.drop('aggregated_features', axis=1) 
# Save the updated DataFrame
df.to_csv(output_file_path, index=False)

# Final message
print(f"Updated dataset saved as: {output_file_path}")
print("Execution finished.")
#%%
import requests
import pandas as pd
import time
import os

# Load your DataFrame
file_path = "/data/saiful/ePPI/alphafold_eppi_embeddings/eppi_alphafold_aggregated_embeddings.csv"
df = pd.read_csv(file_path)

# Define the parent directory and output file path
parent_dir = os.path.dirname(os.path.dirname(file_path))
output_file_path = os.path.join(parent_dir, "eppi_alphafold_updated_embeddings.csv")

# Check if the output file exists to resume progress
if os.path.exists(output_file_path):
    df_existing = pd.read_csv(output_file_path)
    processed_proteins = set(df_existing['protein_id'])  # Track already processed proteins
else:
    df_existing = pd.DataFrame(columns=['protein_id', 'Enzyme'])  # Initialize if no file exists
    processed_proteins = set()

def check_uniprot(protein_id):
    """Check if a protein is an enzyme by querying UniProt."""
    url = f"https://www.uniprot.org/uniprot/{protein_id}.txt"
    response = requests.get(url)

    if response.status_code == 404:
        return "Not Found"
    elif response.status_code == 200:
        return True if "EC=" in response.text else False
    else:
        return "Error"

# Initialize counters and timer
start_time = time.time()
batch_size = 500
processed_count = 0
new_results = []  # Store new results before appending to CSV

# Process proteins and append results
for idx, row in df.iterrows():
    protein_id = row['protein_id']

    # Skip if already processed
    if protein_id in processed_proteins:
        continue

    # Check UniProt for the current protein
    enzyme_status = check_uniprot(protein_id)
    new_results.append([protein_id, enzyme_status])
    processed_count += 1

    # Save results in batches
    if processed_count % batch_size == 0:
        elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
        new_df = pd.DataFrame(new_results, columns=['protein_id', 'Enzyme'])
        new_df.to_csv(output_file_path, mode='a', index=False, header=not os.path.exists(output_file_path))  # Append mode
        processed_proteins.update(new_df['protein_id'])  # Update processed proteins set
        new_results.clear()  # Clear results buffer

        print(f"Processed {processed_count} proteins. Last Protein ID: {protein_id} | Time taken: {elapsed_time:.2f} minutes | Saved to CSV")
        start_time = time.time()  # Reset timer

# Final save after the last batch
if new_results:
    new_df = pd.DataFrame(new_results, columns=['protein_id', 'Enzyme'])
    new_df.to_csv(output_file_path, mode='a', index=False, header=not os.path.exists(output_file_path))  # Append mode

print(f"Final dataset saved as: {output_file_path}")
print("Execution finished.")

