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
    ...
    # 关于本代码及关联论文的贡献声明

本项目所包含的代码及与之直接相关的学术论文《通道注意力指导全局局部语义协同的表情识别》，其全部实质性贡献均由本人（高硕）独立完成。

## 具体贡献细节

本人确认，本人独立完成了以下所有核心工作：
*   论文的核心创新点提出与理论框架构建
*   研究方案设计与实验方法实现
*   代码的编写、测试与调试
*   实验数据的采集、处理与分析
*   论文原稿的撰写与修改

## 关于其他署名方的说明

本论文的其他署名作者（吕景刚 李玉芝 周金）确认，他们未对上述实质性工作做出创造性贡献。其署名仅为满足部分期刊或机构对论文发表流程的形式要求。

## 知识产权与授权

任何个人或实体通过下载、复制、分发或其他方式使用本仓库中的代码，即视为**知晓并同意**本贡献声明之全部内容，认可本人作为本论文及关联代码的唯一实质性贡献者。未经本人书面许可，任何一方不得将本论文或代码用于申请专利、申报奖项或任何可能损害本人权益的用途。

## 生效与确认

本声明自首次提交至本代码仓库之日起生效。
## Prepare dataset
```bash
pip install -r requirements.txt
