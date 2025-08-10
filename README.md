# SKAM-ResistNet — TEM1 Target Finder

**Live Demo:** <https://6135d67466b5563897.gradio.live/>  

---

## Project Description
SKAM-ResistNet is a lightweight deep learning application for predicting **small-molecule binding affinity** to **TEM-1 β-lactamase**, an enzyme responsible for antibiotic resistance in bacteria.

The tool:
- Takes a set of chemical compounds in **SMILES** format.
- Predicts binding affinity (**pAff**, −log10 Kd in molar).
- Outputs a calibrated **binder probability**.
- Generates visual plots to interpret predictions.

This project was developed for the **HackNation Global AI Hackathon** by Team **SKAM**.

---

## Features
- Accepts user-provided SMILES strings.
- Predicts binding affinity to TEM-1 β-lactamase.
- Provides binder probability with confidence intervals.
- Generates bar, scatter, and heatmap visualizations.
- Lightweight, fast, and accessible for small-scale labs.

---

## Setup Instructions

### 1) Clone this repository
### 2) Install dependencies (requirements.txt)
### 3) (Optional) Add dataset
### 4) Run the application

---

## Dependencies
Core packages used:
pandas
numpy
matplotlib
torch
rdkit
gradio
scikit-learn
xgboost
transformers
requests
See requirements.txt for the full list.

---

## Dataset
Source: [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp)
[Google Drive Dataset Link](https://drive.google.com/drive/folders/1vC7mmuokrbXQBr_wZezRtIUhyScAy_1w?usp=sharing)
Organization in Google Drive:
dataset/
  raw/bindingdb_tem1_raw.csv        # Raw BindingDB export
  processed/tem1_clean.csv          # Cleaned dataset used for model training
  README.txt                        # Describes columns and preprocessing

---

## Team SKAM
- **S**amhitha **K**unadharaju
- **A**diti **M**od
