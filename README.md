# CAFA-6 Protein Function Prediction

Kaggle competition solution for predicting protein functions using Gene Ontology (GO) terms based on amino acid sequences.

**Competition**: [CAFA 6 - Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)

## Problem Statement

Given a protein's amino acid sequence, predict which Gene Ontology (GO) terms describe its function across three categories:
- **C** (Cellular Component): Where the protein is located
- **F** (Molecular Function): What the protein does at molecular level
- **P** (Biological Process): What biological processes the protein participates in

This is a **multi-label classification** problem with 26,125 possible GO term labels.

## Dataset Overview

- **82,404 proteins** in training set
- **26,125 unique GO terms** (labels)
- **537,027 total annotations**
- **Average 6.52 labels per protein** (range: 1-233)
- **Sequence lengths**: 3 to 35,213 amino acids (avg: 526 AA)

**Key Challenge**: Extreme class imbalance - 51.6% of labels appear in fewer than 10 proteins!

## Project Structure

```
cafa6/
├── README.md                          # This file
├── CLAUDE.md                          # Detailed EDA findings and model approach
├── eda.py                             # Basic exploratory data analysis
├── eda_advanced.py                    # Advanced EDA with aspect-specific analysis
├── cafa-6-protein-function-prediction/ # Dataset (not in git)
│   ├── Train/
│   │   ├── train_sequences.fasta      # Protein sequences
│   │   ├── train_terms.tsv            # GO term annotations
│   │   ├── train_taxonomy.tsv         # Taxonomy information
│   │   └── go-basic.obo               # GO hierarchy
│   ├── Test/
│   │   ├── testsuperset.fasta         # Test sequences
│   │   └── testsuperset-taxon-list.tsv
│   ├── IA.tsv                         # Information Accretion scores
│   └── sample_submission.tsv          # Submission format
├── notebooks/                         # Jupyter notebooks (TBD)
├── src/                               # Source code (TBD)
└── models/                            # Saved models (TBD)
```

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd cafa6
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio  # PyTorch
pip install transformers  # For protein language models
pip install fair-esm  # ESM-2 model
pip install biopython  # For sequence handling
pip install kaggle  # Kaggle CLI
```

### 4. Download Dataset

#### Option A: Using Kaggle CLI (Recommended)

1. **Setup Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **Download dataset**:
```bash
# Make sure you're in the project directory
cd /path/to/cafa6

# Download and unzip dataset
kaggle competitions download -c cafa-6-protein-function-prediction
unzip cafa-6-protein-function-prediction.zip -d cafa-6-protein-function-prediction
rm cafa-6-protein-function-prediction.zip
```

#### Option B: Manual Download

1. Go to https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data
2. Download the dataset
3. Unzip to `cafa-6-protein-function-prediction/` directory

### 5. Verify Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Run basic EDA
python eda.py

# Run advanced EDA
python eda_advanced.py
```

## Quick Start Guide

### Explore the Data

```bash
# Basic EDA with visualizations
python eda.py

# Advanced aspect-specific analysis
python eda_advanced.py
```

### Read Analysis

See [CLAUDE.md](CLAUDE.md) for:
- Detailed EDA findings
- Class imbalance analysis
- Recommended model architectures
- Implementation roadmap
- Training strategies

## Recommended Approach

Based on EDA findings, the recommended approach is:

1. **Use Protein Language Models** (ESM-2, ProtBERT)
2. **Multi-task architecture** with separate heads for C/F/P aspects
3. **Focal loss** to handle extreme class imbalance
4. **Ensemble** multiple models for robustness
5. **Threshold optimization** per label

See [CLAUDE.md](CLAUDE.md) for detailed architecture and training strategy.

## Key Insights from EDA

1. **Extreme imbalance**: 51.6% of labels appear in < 10 proteins
2. **Multi-aspect patterns**: 44.6% of proteins have all three aspects (C+F+P)
3. **Complexity varies**: P (16,858 labels) >> F (6,616) > C (2,651)
4. **Dominant label**: GO:0005515 (protein binding) in 40.9% of proteins
5. **Variable lengths**: Sequences range from 3 to 35,213 amino acids

## Development Roadmap

- [x] Initial data exploration
- [x] Advanced EDA and analysis
- [ ] Baseline model (Dense NN + ESM-2 embeddings)
- [ ] Multi-task architecture implementation
- [ ] Focal loss for imbalance handling
- [ ] Model ensemble
- [ ] Threshold optimization
- [ ] Final submission

## Resources

- **Competition**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- **ESM-2 Model**: https://github.com/facebookresearch/esm
- **Gene Ontology**: http://geneontology.org/
- **DeepGOPlus**: https://github.com/bio-ontology-research-group/deepgoplus

## License

This project is for educational and competition purposes.

## Contact

For questions or collaboration, please open an issue or reach out via Kaggle.

---

**Note**: This is a work in progress. Check [CLAUDE.md](CLAUDE.md) for detailed analysis and modeling approach.
