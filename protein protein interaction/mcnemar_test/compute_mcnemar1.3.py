#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 07:20:48 2025

@author: saiful
"""
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import math
from itertools import combinations
import sys
model_name  = "GAT"
sys.stdout = open(f"../mcnemar_tests/p_values/mcnemar_output_{model_name}.txt", "w")

def perform_mcnemar_test(embeddingA_name, embeddingA_path, embeddingB_name, embeddingB_path):
    # Load .npz files
    dataA = np.load(embeddingA_path)
    dataB = np.load(embeddingB_path)

    # Extract arrays
    y_trueA = dataA['y_true']
    y_predA = dataA['y_pred']
    y_trueB = dataB['y_true']
    y_predB = dataB['y_pred']

    # Check if ground truths match
    if not np.array_equal(y_trueA, y_trueB):
        raise ValueError(f"Mismatch: y_true arrays from {embeddingA_name} and {embeddingB_name} are not identical.")
    else:
        print(f"\ny_true arrays match for {embeddingA_name} and {embeddingB_name}.")

    # Create contingency table for McNemar's test
    correctA = (y_predA == y_trueA)
    correctB = (y_predB == y_trueA)

    # Contingency table:
    # - a: both models correct
    # - b: embeddingA correct, embeddingB incorrect
    # - c: embeddingA incorrect, embeddingB correct
    # - d: both models incorrect
    a = np.sum(correctA & correctB)
    b = np.sum(correctA & ~correctB)
    c = np.sum(~correctA & correctB)
    d = np.sum(~correctA & ~correctB)

    # Create 2x2 contingency table
    contingency_table = np.array([[a, b], [c, d]])
    print(f"\nContingency Table for {embeddingA_name} vs {embeddingB_name}:")
    print(f"Both correct (a): {a}")
    print(f"{embeddingA_name} correct, {embeddingB_name} incorrect (b): {b}")
    print(f"{embeddingA_name} incorrect, {embeddingB_name} correct (c): {c}")
    print(f"Both incorrect (d): {d}")

    # Consider effect size calculation
    effect_size = abs(b - c) / (b + c) if (b + c) > 0 else 0
    print(f"Effect size (|b-c|/(b+c)): {effect_size:.4f}")

    # Perform McNemar's test
    # Add this check for the exact test condition
    if (b + c) < 25:
        result = mcnemar(contingency_table, exact=True)
        print("Using exact test due to small discordant pairs")
    else:
        result = mcnemar(contingency_table, exact=False, correction=True)
        print("Using chi-squared approximation with continuity correction")
    # result = mcnemar(contingency_table, exact=False, correction=True)  # Use chi-squared approximation
    statistic = result.statistic
    p_value = result.pvalue

    # Print results
    print(f"\nMcNemar's Test Results for {embeddingA_name} vs {embeddingB_name}:")
    print(f"Statistic: {statistic:.20f}")
    print(f"P-value: {p_value:.20f}")
    print(f"Total Test Cases: {len(y_trueA)}")


    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: There is a significant difference between {embeddingA_name} and {embeddingB_name} predictions.")
    else:
        print(f"Result: No significant difference between {embeddingA_name} and {embeddingB_name} predictions.")

# Define embedding names and paths
embeddings = {
    "AlphaFold": "/home/saiful/ePPI_dgl/mcnemar_tests/gat_predictions/alphafold_feat_gat_lr_1e-3_checkpoint_y_true_y_pred.npz",
    "ESM": "/home/saiful/ePPI_dgl/mcnemar_tests/gat_predictions/esm_features_gat_lr_1e-4_checkpoint_y_true_y_pred.npz",
    "BioEmbedding": "/home/saiful/ePPI_dgl/mcnemar_tests/gat_predictions/bioembed_feat_gat_lr_1e-4_checkpoint_r1_y_true_y_pred.npz",
    "ProtVec": "/home/saiful/ePPI_dgl/mcnemar_tests/gat_predictions/protvec_feat_gat_lr_1e-4_checkpoint_r1_y_true_y_pred.npz"
}
# Example usage: Iterate over all unique pairs
if __name__ == "__main__":
    # Generate all unique pairs of embeddings
    embedding_pairs = list(combinations(embeddings.keys(), 2))
    print(f"Results for the model  {model_name}")

    print(f"Total number of pairwise comparisons: {len(embedding_pairs)}\n")
    
    # Perform McNemar's test for each pair
    for embeddingA_name, embeddingB_name in embedding_pairs:
        embeddingA_path = embeddings[embeddingA_name]
        embeddingB_path = embeddings[embeddingB_name]
        print("=" * 80)
        print(f"Comparing {embeddingA_name} vs {embeddingB_name}")
        print("=" * 80)
        perform_mcnemar_test(embeddingA_name, embeddingA_path, embeddingB_name, embeddingB_path)

        print("\n\n")