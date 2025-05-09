#!/bin/bash
# ./run_alphafold.sh

# Export the GPU to use 
export CUDA_VISIBLE_DEVICES=7

## === ##
# Enable unified memory and dynamic allocation for JAX
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use 4 GPUs
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export TF_FORCE_UNIFIED_MEMORY=1
# export XLA_PYTHON_CLIENT_MEM_FRACTION=9.0
# Limit GPU memory usage to 90% 
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9


# Use multiple GPUs

## === ##


# Redirect stdout and stderr to a file
# exec > split_output8.txt 2>&1
# exec > ./split_output_txt/split_output_gpu7.txt 2>&1
exec > /data/saiful/ePPI/alphafold_eppi_embeddings/console_output/console_output_batch_x.txt 2>&1



# Start time
start_time=$(date +%s)

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alphafold38_gpu3

# Set the path to the sequences file
# SEQUENCES_FILE="./prot_files/sequences_converted.txt"
#SEQUENCES_FILE="/data/saiful/ePPI/prot_to_fasta_all.txt"
#SEQUENCES_FILE="/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold/batch_9.txt"
SEQUENCES_FILE="/data/saiful/ePPI/prot_to_fasta_splits_for_alphafold_remaining/batch_x.txt"




# Debug print statement
echo "Attempting to create directory ./results/embeddings"

# Create a directory to store the individual pickle files if it doesn't exist
mkdir -p ./results/embeddings

# Check if the directory creation was successful
if [ -d "./results/embeddings" ]; then
  echo "Directory ./results/embeddings created successfully."
else
  echo "Failed to create directory ./results/embeddings."
  exit 1
fi

# Initialize a counter
counter=0

# Read sequences from the file and process them one by one
while read -r line; do

  echo -e "\n\n  ==== ==== ==== ==== start ==== ==== ==== ===="
  echo "Processing line: $line"
  if [[ $line == ">"* ]]; then
    # Extract protein ID
    PROTEIN_ID=$(echo $line | cut -d '>' -f 2)
    # echo "Protein ID: $PROTEIN_ID"
        counter=$((counter + 1))
    echo "Protein ID No. [$counter]: $PROTEIN_ID"
  else
    # Extract sequence
    SEQUENCE=$line
    echo "Sequence: $SEQUENCE"

    # flag 1.41 Write individual sequence to a temporary file
    TEMP_FASTA_FILE="./temp_sequence.fasta"
    echo ">$PROTEIN_ID" > $TEMP_FASTA_FILE
    echo $SEQUENCE >> $TEMP_FASTA_FILE
    echo "Temporary fasta file created: $TEMP_FASTA_FILE"

    # Run the Python script with the current sequence
    python3 Representations_AlphaFold2_v3.23.py $TEMP_FASTA_FILE $PROTEIN_ID
    # python3 Representations_AlphaFold2_v3.25.py $TEMP_FASTA_FILE $PROTEIN_ID

    echo "Ran Python script for: $PROTEIN_ID"

    # Remove the temporary file
    rm $TEMP_FASTA_FILE
    echo "Temporary fasta file removed: $TEMP_FASTA_FILE"
    echo -e "=========     ==    finish    ==      =========="
  fi
done < $SEQUENCES_FILE


# End time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Convert execution time to hours, minutes, and seconds
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

# Print the total execution time
echo "Total execution time (all proteins): $hours hours, $minutes minutes, and $seconds seconds"

