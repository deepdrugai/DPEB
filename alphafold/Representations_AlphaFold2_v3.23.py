#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:18:26 2024
This file saves the embeddings in individual pickle files
@author: saiful
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
# sys.stdout = open("console_output.txt", "w")
import time
start_time = time.time()
import numpy as np
np.int = int
np.object = object
np.bool = np.bool_
np.typeDict = np.sctypeDict

import time
import sys
import os
import re
import hashlib
import pickle
from absl import logging
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
from string import ascii_uppercase
import tensorflow as tf
import jax
import jax.numpy as jnp
import gc

import jax

devices = jax.devices()
# Print the device list
for device in devices:
    print(device)
# Import functions from external script
from my_functions import set_bfactor, mk_template, mk_mock_template

# Check if GPU is available and enable memory growth
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def add_hash(x, y):
    return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]


def predict_structure(prefix, feature_dict, Ls, model_params, use_model, do_relax=False, random_seed=0):
    """Predicts structure using AlphaFold for the given sequence."""
    idx_res = feature_dict['residue_index']
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += 200
        L_prev += L_i
    chains = list("".join([ascii_uppercase[n]*L for n, L in enumerate(Ls)]))
    feature_dict['residue_index'] = idx_res

    plddts, paes = [], []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []

    for model_name, params in model_params.items():
        if model_name in use_model:
            print(f"running {model_name}")
            if any(str(m) in model_name for m in [1, 2]):
                model_runner = model_runner_1
            if any(str(m) in model_name for m in [3, 4, 5]):
                model_runner = model_runner_3
            model_runner.params = params

            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
            prediction_result = model_runner.predict(processed_feature_dict)
            unrelaxed_protein = protein.from_prediction(processed_feature_dict, prediction_result)
            unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
            plddts.append(prediction_result['plddt'])
            paes.append(prediction_result['predicted_aligned_error'])

            if do_relax:
                amber_relaxer = relax.AmberRelaxation(max_iterations=0, tolerance=2.39,
                                                      stiffness=10.0, exclude_residues=[],
                                                      max_outer_iterations=20)
                relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
                relaxed_pdb_lines.append(relaxed_pdb_str)

    lddt_rank = np.mean(plddts, -1).argsort()[::-1]
    out = {}
    print("reranking models based on avg. predicted lDDT")
    for n, r in enumerate(lddt_rank):
        print(f"model_{n+1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f'./results/pdb/{prefix}_unrelaxed_model_{n+1}.pdb'
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdb_lines[r])
        set_bfactor(unrelaxed_pdb_path, plddts[r]/100, idx_res, chains)

        if do_relax:
            relaxed_pdb_path = f'./results/pdb/{prefix}_relaxed_model_{n+1}.pdb'
            with open(relaxed_pdb_path, 'w') as f:
                f.write(relaxed_pdb_lines[r])
            set_bfactor(relaxed_pdb_path, plddts[r]/100, idx_res, chains)

        out[f"model_{n+1}"] = {"plddt": plddts[r], "pae": paes[r]}
    print("predict_structure() finished")
    return out, prediction_result

def read_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        protein_id = lines[0].strip().replace('>', '')
        sequence = ''.join([line.strip() for line in lines[1:]])
    return protein_id, sequence

if __name__ == "__main__":
    
    # Command-line arguments
    # protein_id = sys.argv[1]
    # sequence = sys.argv[2]
    
    # Read protein_id and sequence from temp_sequence.fasta file
    fasta_file = sys.argv[1]
    protein_id, sequence = read_fasta(fasta_file)
    
    print("flag 1.11 protein_id : ", protein_id)
    print("flag 1.11 sequence : ", sequence)


    # Dictionary to store results
    results = {}

    print("\n\n## running protein ID : ", protein_id)
    query_sequence = re.sub(r'[^a-zA-Z]', '', sequence).upper()
    jobname = add_hash(protein_id, query_sequence)

    msa_mode = "custom"  #"single_sequence", "custom" 
    num_models = 1
    use_msa = False
    use_env = False
    use_custom_msa = False
    use_amber = False
    use_templates = True
    homooligomer = 1
    
    print("# num_models :", num_models)

    if homooligomer > 1:
        if use_amber:
            print("amber disabled: amber is not currently supported for homooligomers")
            use_amber = False
        if use_templates:
            print("templates disabled: templates are not currently supported for homooligomers")
            use_templates = False

    # Setup the model
    if "model" not in dir():
        import warnings
        from absl import logging
        warnings.filterwarnings('ignore')
        logging.set_verbosity("error")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import Bio
        import py3Dmol
        import matplotlib.pyplot as plt
        import ipywidgets
        from ipywidgets import interact, fixed, GridspecLayout, Output

    if use_amber and "relax" not in dir():
        sys.path.insert(0, '/usr/local/lib/python3.7/site-packages/')
        from alphafold.relax import relax

    # Collect model weights
    use_model = {}
    if "model_params" not in dir():
        model_params = {}
    for model_name in ["model_1", "model_2", "model_3", "model_4", "model_5"][:num_models]:
        use_model[model_name] = True
        if model_name not in model_params:
            model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=".")
            if model_name == "model_1":
                model_config = config.model_config(model_name+"_ptm")
                model_config.data.eval.num_ensemble = 1
                model_runner_1 = model.RunModel(model_config, model_params[model_name])
            if model_name == "model_3":
                model_config = config.model_config(model_name+"_ptm")
                model_config.data.eval.num_ensemble = 1
                model_runner_3 = model.RunModel(model_config, model_params[model_name])

    # Parse templates
    if use_templates and os.path.isfile(f"{jobname}_hhm.ffindex"):
        template_features = mk_template(jobname, query_sequence)
    else:
        template_features = mk_mock_template(query_sequence * homooligomer)

    # Parse MSA
    a3m_file = f"./results/a3m/{jobname}.single_sequence.a3m"
    with open(a3m_file, "w") as text_file:
        text_file.write(f">1\n{query_sequence}")

    a3m_lines = "".join(open(a3m_file, "r").readlines())
    msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

    if homooligomer == 1:
        msas = [msa]
        deletion_matrices = [deletion_matrix]
    else:
        msas = []
        deletion_matrices = []
        Ln = len(query_sequence)
        for o in range(homooligomer):
            L = Ln * o
            R = Ln * (homooligomer-(o+1))
            msas.append(["-"*L+seq+"-"*R for seq in msa])
            deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])

    # Gather features
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence*homooligomer,
                                          description="none",
                                          num_res=len(query_sequence)*homooligomer),
        **pipeline.make_msa_features(msas=msas, deletion_matrices=deletion_matrices),
        **template_features
    }

    # Predict structure
    outs, prediction_result = predict_structure(jobname, feature_dict,
                             Ls=[len(query_sequence)]*homooligomer,
                             model_params=model_params, use_model=use_model,
                             do_relax=use_amber)

    # Extract embeddings and save to results dictionary
    embeddings = prediction_result['representations']['single']
    results[protein_id] = {
        'protein_id': protein_id,
        'fasta': query_sequence,
        'embedding': embeddings
    }

    # Save the results dictionary as a pickle file
    # with open(f'./results/embeddings/split_embeddings/{protein_id}_embedding.pkl', 'wb') as handle:
    with open(f'/data/saiful/ePPI/alphafold_eppi_embeddings/batch_x/{protein_id}_embedding.pkl', 'wb') as handle:
        pickle.dump(results[protein_id], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Clear sessions and garbage collect
    tf.keras.backend.clear_session()
    jax.clear_backends()
    gc.collect()

    print(f"Processing of protein ID {protein_id} finished.")
    
    end_time = time.time()
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f} (hours:minutes:seconds)")
    print("execution finished..")
