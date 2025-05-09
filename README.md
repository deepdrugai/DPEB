# 🧬 DPEB: Deep Protein Embedding Benchmark Environment

This repository provides the environment setup used in our experiments related to deep protein embeddings, graph neural networks, and AlphaFold-based representations.

---

## 📦 Environment Setup

To reproduce the results or run any experiments in this repository, use the provided Conda environment file to set up your environment. The environment is named `DPEB` and includes dependencies such as PyTorch, DGL (CUDA 10.2), scikit-learn, transformers, and AlphaFold-related utilities.

### 🔗 Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA 10.2-compatible GPU (for using GPU-based operations via PyTorch + DGL)

---

### 🚀 Create the Environment

You can recreate the `DPEB` environment using the provided `env.yml` file:

```bash
conda env create -f /home/saiful/DPEB/env.yml
```

###  ✅ Step 3: Activate the Environment
```bash
conda activate DPEB
```

💡 Usage Notes
After activating the environment, you're ready to:

Run AlphaFold embedding scripts

Use DGL-based GNNs for protein-protein interaction prediction

Load and analyze protein sequence datasets

Train transformer-based or hybrid models for biological tasks
