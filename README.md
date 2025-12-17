# Geometric-aware Deep Learning for Deciphering Tissue Structure from Spatially Resolved Transcriptomics

![SpatialGEO](SpatialGEO/SpatialGEO_workflow.png) 

---

## Overview

SpatialGEO is a geometric-aware deep learning framework designed to dissect complex tissue structures from Spatially Resolved Transcriptomics (SRT) data across diverse platforms and resolutions, ranging from spot-level to single-cell and sub-cellular scales. The framework initially employs a dual-encoder architecture to extract embeddings, utilizing an Autoencoder (AE) to capture intrinsic gene expression features and an Enhanced Graph Autoencoder (EGAE) to reinforce spatial graph representations by simultaneously reconstructing gene attributes and spatial adjacency. To further refine these representations, SpatialGEO incorporates a geometric-aware latent embedding generation module that dynamically fuses cross-modal potentials. By leveraging geometric graph learning to characterize continuous spatial relationships between spots, this module effectively overcomes the limitations of binary adjacency graphs in modeling spatial continuity. The entire framework is optimized via a triplet self-supervised strategy, which unifies the learning objectives of the AE, EGAE, and the fusion module. This unified paradigm enhances the alignment between modalities and promotes efficient information integration, yielding robust embeddings that support a wide range of downstream analysis tasks.

---

## Requirements

Clone this repository:

```bash
git clone https://github.com/xingyili/SpatialGEO.git
cd SpatialGEO
```
Install the required core dependencies (preferably in a virtual environment):

- `python` 3.10.13
- `h5py` 3.12.1
- `igraph` 0.11.8
- `munkres` 1.1.4
- `numpy` 2.2.4
- `pandas` 2.2.3
- `rpy2` 3.5.17
- `scanpy` 1.10.4
- `scikit-learn` 1.6.0
- `scipy` 1.14.1
- `seaborn` 0.13.2
- `torch` 2.5.1+cu118
- `torch_cluster` 1.6.3+pt25cu118
- `torch-geometric` 2.6.1
- `torch_scatter` 2.1.2+pt25cu118
- `torch_sparse` 0.6.18+pt25cu118
- `torchaudio` 2.5.1+cu118
- `torchvision` 0.20.1+cu118
- `tqdm` 4.67.1

## **Model Training**
### **1. Run Full Pipeline**
We provide an automated shell script to run the full training pipeline (AE → GAE → Pretrain → SpatialGEO) with a single command.

To train the model on the demo dataset, simply run:
```bash
# 1. Grant execution permission (only needed once)
chmod +x run_pipeline.sh

# 2. Run the automated pipeline
./run_pipeline.sh
```
### **2. Step-by-Step Training**
If you prefer to run each module individually for debugging or parameter adjustment, please follow the order below strictly, as each step depends on the output of the previous one.

Step 1: Train Autoencoder (AE) Extracts high-level features from the gene expression matrix.
```bash
cd AE
python main_new.py
cd ..
```
Step 2: Train Graph Autoencoder (GAE) Captures spatial topological structures and generates graph embeddings.
```bash
cd GAE
python main_new.py
cd ..
```
Step 3: Pretraining Aligns the feature spaces of the AE and GAE modules.
```bash
cd Pretrain
python main_new.py
cd ..
```
Step 4: SpatialGEO (Main Model) Performs the final training, latent embedding fusion, and clustering.
```bash
cd spatialgeo
python main_new.py
cd ..
```
# SpatialGEO
