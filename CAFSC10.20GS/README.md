# CAFSC Network (Public Review Version)

This repository provides a runnable, public review version of **CAFSC** (Collaborative Adaptive Framework for Noisy Facial Expression Recognition).

**Protection level:** Light — The code runs with default/simplified parameters, but some internal coefficients are intentionally omitted to prevent exact replication of reported results.

---

## Contents
- `train.py` — main training script  
- `model/` — model implementation (CAFSC_Net, CAM-ACR, Local-Global Enhancement, CAFM)  
- `data/augmentation.py` — anti-noise data augmentation  
- `utils/` — training utilities and plotting  
- `requirements.txt` — Python dependencies  

---

## Experimental Setup

All experiments were conducted on the following hardware and software configuration:

- **GPU:** NVIDIA RTX 4080 Ti (16GB)
- **CPU:** Intel Core i9-10885H
- **Memory:** 8 GB RAM  
- **OS:** Windows 10  
- **Python:** 3.10.14 (Conda environment)  
- **PyTorch:** 2.5.1 + CUDA 12.1  
- **Torchvision:** 0.16.2  

**Training Configuration:**
- **Backbone:** Pre-trained ResNet-50 convolutional layers (from PyTorch)
- **Optimizer:** Adam  
- **Initial Learning Rate:** 0.0005  
- **Scheduler:** Cosine Annealing LR  
- **Loss Function:** Cross-Entropy Loss  
- **Epochs:** 50  
- **Data Loading:** 4 worker threads with `pin_memory=True`  
- **Framework:** CAFSC built on ResNet50, incorporating a Global–Local Feature Collaborative Fusion module  
- **Classifier:** Softmax  

---

## Environment Setup

Create the environment using Conda or Python virtual environment:

```bash
conda create -n cafsc python=3.10
conda activate cafsc
pip install -r requirements.txt

## Contents
- `train.py` — main training script
- `model/` — model implementation (CAFSC_Net, CAM-ACR, Local-Global Enhancement, CAFM)
- `data/augmentation.py` — anti-noise data augmentation
- `utils/` — training utilities and plotting
- `requirements.txt` — Python dependencies

## Quickstart
1. Install dependencies:
## Prepare dataset
RAF/
├── train/
│   ├── class0/
│   ├── class1/
│   ...
└── valid/
    ├── class0/
    ├── class1/
Statement on Contributions to This Codebase and the Associated Paper

The code contained in this repository and the directly related academic paper, Channel Attention-Guided Global–Local Semantic Collaboration for Facial Expression Recognition, were completed primarily by Shuo Gao.

Detailed Contributions

```bash
pip install -r requirements.txt
