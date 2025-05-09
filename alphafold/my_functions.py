#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:18:26 2024

@author: saiful
"""

# -*- coding: utf-8 -*-

import os
import os.path
import re
import hashlib
import time
import fileinput
import sys
start_time = time.time()

import numpy as np
print("np.__version__ :" ,np.__version__)

import warnings
from absl import logging
import os

import Bio
print("Bio.__version__)", Bio.__version__)

# pip install biopython==1.76
# pip install dm-haiku==0.0.5
# Alias np.int to int
import numpy as np

np.int = int
np.object = object
np.bool = np.bool_
np.typeDict = np.sctypeDict

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data.tools import hhsearch

from alphafold.model import config
from alphafold.model import data
from alphafold.model import model

import py3Dmol
import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import interact, fixed, GridspecLayout, Output
from alphafold.relax import relax
from string import ascii_uppercase

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check if GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU is available")
    for gpu in gpu_devices:
        print(gpu)
else:
    print("GPU is not available")

def add_hash(x,y):
  print("add_hash() called.. ")
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]


def mk_mock_template(query_sequence):
  print("mk_mock_template() called.. ")

  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def mk_template(jobname, query_sequence):
  print("mk_template() called.. ")

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir="templates/",
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None)

  hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch",databases=[jobname])

  a3m_lines = "\n".join(open(f"{jobname}.a3m","r").readlines())
  hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                       query_pdb_code=None,
                                                       query_release_date=None,
                                                       hits=hhsearch_hits)
  return templates_result.features

def set_bfactor(pdb_filename, bfac, idx_res, chains):
  print("set_bfactor() called.. ")

  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()



#%%
# #@title Make plots

# ##################################################################
# plt.figure(figsize=(12,4),dpi=100)
# plt.subplot(1,2,1); plt.title("Number of sequences per position")
# plt.plot((feature_dict["msa"] != 21).sum(0))
# plt.xlabel("positions")
# plt.ylabel("number of sequences")
# plt.subplot(1,2,2); plt.title("Predicted lDDT per position")
# for model_name,value in outs.items():
#   plt.plot(value["plddt"],label=model_name)
# if homooligomer > 0:
#   for n in range(homooligomer+1):
#     x = n*(len(query_sequence)-1)
#     plt.plot([x,x],[0,100],color="black")
# plt.legend()
# plt.ylim(0,100)
# plt.ylabel("predicted lDDT")
# plt.xlabel("positions")
# plt.savefig(jobname+"_coverage_lDDT.png")
# plt.show()

# print("Predicted Alignment Error")
# ##################################################################
# plt.figure(figsize=(3*num_models,2), dpi=100)
# for n,(model_name,value) in enumerate(outs.items()):
#   plt.subplot(1,num_models,n+1)
#   plt.title(model_name)
#   plt.imshow(value["pae"],label=model_name,cmap="bwr",vmin=0,vmax=30)
#   plt.colorbar()
# plt.savefig(jobname+"_PAE.png")
# plt.show()
# ##################################################################

#@title Display 3D structure

def plot_plddt_legend():
  print("plot_plddt_legend() called.. ")

  thresh = ['plDDT:','Very low (<50)','Low (60)','OK (70)','Confident (80)','Very high (>90)']
  plt.figure(figsize=(1,0.1),dpi=100)
  ########################################
  for c in ["#FFFFFF","#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF"]:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5)
  plt.axis(False)
  return plt

def plot_confidence(model_name, outs, homooligomer, query_sequence):
  print("plot_confidence() called.. ")

  plt.figure(figsize=(6, 10))
  """Plots the legend for plDDT."""
  #########################################
  plt.subplot(2,1,1); plt.title('Predicted lDDT')
  plt.plot(outs[model_name]["plddt"])
  for n in range(homooligomer+1):
    x = n*(len(query_sequence))
    plt.plot([x,x],[0,100],color="black")
  plt.xlabel('Residue')
  plt.ylabel('plDDT')
  #########################################
  plt.subplot(2,1,2); plt.title('Predicted Aligned Error')
  plt.imshow(outs[model_name]["pae"], cmap="bwr",vmin=0,vmax=30)
  plt.colorbar()
  plt.xlabel('Scored residue')
  plt.ylabel('Aligned residue')
  #########################################
  return plt

def show_pdb(model_name, show_sidechains, show_mainchain, color, use_amber, jobname, homooligomer):
  print("show_pdb() called.. ")

  if use_amber:
    pdb_filename = f"{jobname}_relaxed_{model_name}.pdb"
  else:
    pdb_filename = f"{jobname}_unrelaxed_{model_name}.pdb"

  view = py3Dmol.view(width=800, height=600)
  view.addModel(open(pdb_filename,'r').read(),'pdb')
  if color == "lDDT":
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}})
  elif color == "rainbow":
    view.setStyle({'cartoon': {'color':'spectrum'}})
  elif color == "chain":
    for n,chain,color in zip(range(homooligomer),list("ABCDEFGH"),
                     ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]):
       view.setStyle({'chain':chain},{'cartoon': {'color':color}})
  if show_sidechains:
    BB = ['C','O','N']
    view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
  if show_mainchain:
    BB = ['C','O','N','CA']
    view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

  view.zoomTo()
  if color == "lDDT":
    grid = GridspecLayout(6, 2)
    out = Output()
    with out: view.show()
    grid[:-1,0] = out
    out = Output()
    with out: plot_confidence(model_name).show()
    grid[:,1] = out
    out = Output()
    with out: plot_plddt_legend().show()
    grid[-1,0] = out
    return grid
  else:
    return view.show()



def predict_structure2(prefix, feature_dict, Ls, model_params, use_model,model_runner_1, model_runner_3, do_relax=False, random_seed=0):
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