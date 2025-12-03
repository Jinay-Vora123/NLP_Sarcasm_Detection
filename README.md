# SarcasmLens: Attention-Based Sarcasm Detection in Hindi–English Code-Mixed Social Media

This repository contains the code for **Subtask 2** and **Subtask 3** of the project:

> **SarcasmLens: Attention-Based Sarcasm Detection in Hindi–English Code-Mixed Social Media**

The goal is to build and analyze models for **binary sarcasm detection** on Hindi–English (Hinglish) code-mixed social media posts. We start from strong classical baselines and then move to neural sequence models with attention.

- **Subtask 2**: Classical ML + BiLSTM baseline  
- **Subtask 3**: Attention-based BiLSTM with interpretability and performance comparison

---

## Dataset

We use the publicly available **HackArena Theme 2: Multilingual Sarcasm Detection** dataset from Kaggle.

- Kaggle dataset link:  
  https://www.kaggle.com/datasets/divyanshu134/hackarena-theme-2-multilingual-sarcasm-detection/data

Core fields we use:

- `Tweet` – the raw text (often Hindi–English code-mixed, sometimes transliterated)
- `Label` – sarcasm label (`YES` = sarcastic, `NO` = non-sarcastic)

You can either:

1. Download the dataset manually from Kaggle and place the files in a `data/` directory, then adjust paths in the notebooks if needed, **or**
2. Use Kaggle’s API to download the dataset programmatically.

---

## Repository Structure

Key files (relevant for this project):

- `subtask2.ipynb`  
  Implements:
  - Data loading and preprocessing (token cleaning, emoji handling, etc.)
  - TF–IDF feature extraction
  - Classical baselines:
    - Logistic Regression
    - Linear SVM
  - Neural baseline:
    - BiLSTM with pre-trained word embeddings
  - Evaluation (accuracy, F1, confusion matrix, basic analysis)

- `subtask3.ipynb`  
  Builds on Subtask 2 and adds:
  - Custom attention layer on top of the BiLSTM encoder
  - BiLSTM + Attention architecture
  - Training with early stopping and learning rate scheduling
  - Quantitative comparison vs. Subtask 2 models
  - Qualitative analysis using attention weights for interpretability

(If you add more scripts or helper modules later, you can list them here.)

---

## Environment and Requirements

You can run the notebooks in a standard Python 3 environment with common ML libraries.

Typical dependencies:

- Python 3.8+
- Jupyter / JupyterLab
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` / `seaborn` (for plots, if used)
- `tensorflow` / `keras` (for BiLSTM and attention models)
- `emoji` (for emoji-to-text conversion)
- `tqdm` (optional, for progress bars)

Example (pip):

```bash
pip install numpy pandas scikit-learn matplotlib emoji tqdm tensorflow
