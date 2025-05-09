#!/bin/bash

USE_AMBER=$1
USE_MSA=$2
USE_TEMPLATES=$3
ENV_NAME="alphafold37"

# Create and activate a new conda environment
if ! conda info --envs | grep -q ${ENV_NAME}; then
  echo "Creating new conda environment: ${ENV_NAME}"
  conda create -y -n ${ENV_NAME} python=3.7
fi

echo "Activating conda environment: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

if [ ! -f AF2_READY ]; then
  echo "Installing dependencies in ${ENV_NAME}..."
  pip install biopython dm-haiku ml-collections py3Dmol

  echo "Downloading model..."
  if [ ! -d "alphafold/" ]; then
    git clone https://github.com/xinformatics/alphafold.git
    mv alphafold alphafold_
    mv alphafold_/alphafold .
    sed -i "s/pdb_lines.append('END')//" alphafold/common/protein.py
    sed -i "s/pdb_lines.append('ENDMDL')//" alphafold/common/protein.py
  fi

  echo "Downloading model params..."
  if [ ! -d "params/" ]; then
    wget https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
    mkdir params
    tar -xf alphafold_params_2021-07-14.tar -C params/
    rm alphafold_params_2021-07-14.tar
  fi
  touch AF2_READY
fi

if [ ${USE_MSA} == "True" ] || [ ${USE_TEMPLATES} == "True" ]; then
  if [ ! -f MMSEQ2_READY ]; then
    echo "Installing MMseqs2 dependencies..."
    sudo apt-get update
    sudo apt-get -y install jq curl zlib1g gawk
    touch MMSEQ2_READY
  fi
fi

if [ ${USE_TEMPLATES} == "True" ] && [ ! -f HH_READY ]; then
  echo "Installing template search tools..."
  conda install -y -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0
  touch HH_READY
fi

if [ ${USE_AMBER} == "True" ] && [ ! -f AMBER_READY ]; then
  echo "Setting up OpenMM for amber refinement..."
  conda install -y -c conda-forge openmm=7.5.1 pdbfixer
  (cd ~/miniconda3/envs/${ENV_NAME}/lib/python3.7/site-packages; patch -s -p0 < /path/to/alphafold_/docker/openmm.patch)
  wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
  mv stereo_chemical_props.txt alphafold/common/
  touch AMBER_READY
fi
