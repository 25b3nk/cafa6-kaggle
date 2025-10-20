# CAFA-6 Protein Function Prediction - Baseline Solution

Complete PyTorch implementation for predicting protein functions using ESM-2 embeddings and multi-task learning.

**Competition**: [CAFA 6 - Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)

---

## Quick Start

### Dataset
- 82,404 proteins with 26,125 GO term labels (multi-label classification)
- Extreme class imbalance: 51.6% of labels appear in <10 proteins
- Three aspects: C (Cellular Component), F (Molecular Function), P (Biological Process)

### Recommended Approach
1. Generate ESM-2 embeddings (~1.5-2 hours on GPU)
2. Train multi-task classifier with focal loss (~30 minutes)
3. Expected F1: 0.45-0.55 (competitive baseline)

### For Kaggle Users (Fastest Path)
**See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)** for complete step-by-step Kaggle notebook workflow.

### For Local Development
**See [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)** for model details, sequence handling, and GPU optimization.

### Want to Understand the Analysis?
**See [CLAUDE.md](CLAUDE.md)** for EDA findings, key decisions, and implementation rationale.

---

## Installation

```bash
git clone <your-repo-url>
cd cafa6
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchvision torchaudio
pip install fair-esm transformers biopython
pip install pandas numpy scikit-learn tqdm
pip install kaggle  # For dataset download
```

---

## Download Dataset

### Using Kaggle CLI (Recommended)

```bash
# Setup Kaggle API (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/kaggle.json
# 4. chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c cafa-6-protein-function-prediction
unzip cafa-6-protein-function-prediction.zip -d cafa-6-protein-function-prediction
rm cafa-6-protein-function-prediction.zip
```

### Manual Download
1. Go to [competition data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data)
2. Download and extract to `cafa-6-protein-function-prediction/`

---

## Usage

### 1. Generate ESM-2 Embeddings

**Local (with GPU):**
```bash
python scripts/generate_embeddings.py \
    --data_dir cafa-6-protein-function-prediction \
    --output_dir data/embeddings \
    --model_name esm2_t30_150M_UR50D \
    --strategy balanced \
    --batch_size 8

# Time: ~1.5-2 hours
# Output: data/embeddings/train_embeddings_esm2_t30_150M_UR50D.pkl (~20 GB)
```

**Kaggle (P100 GPU):**
```python
# See KAGGLE_GUIDE.md for complete Kaggle workflow
from generate_embeddings_kaggle import generate_embeddings_kaggle

embeddings_path = generate_embeddings_kaggle(
    data_dir='/kaggle/input/cafa-6-protein-function-prediction',
    model_name='esm2_t30_150M_UR50D',
    batch_size=4
)
```

### 2. Train Baseline Model

```bash
python scripts/train_baseline.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --epochs 10 \
    --batch_size 32 \
    --loss focal

# Time: ~30 minutes
# Expected F1: 0.50-0.60
```

### 3. Train Multi-Task Model (Better Performance)

```bash
python scripts/train_multitask.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --epochs 15 \
    --learnable_weights

# Time: ~45 minutes
# Expected F1: 0.55-0.65
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --model models/baseline/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --optimize_threshold
```

---

## Project Structure

```
cafa6/
├── README.md                    # This file
├── KAGGLE_GUIDE.md             # Complete Kaggle workflow (15 steps)
├── TECHNICAL_NOTES.md          # Model details, GPU optimization, sequence handling
│
├── scripts/
│   ├── generate_embeddings.py              # Local embedding generation
│   ├── generate_embeddings_kaggle.py       # Kaggle (P100 single GPU)
│   ├── generate_embeddings_kaggle_dual_gpu.py  # Kaggle (T4 x2 dual GPU)
│   ├── train_baseline.py                   # Train simple classifier
│   ├── train_multitask.py                  # Train multi-task model
│   └── evaluate.py                         # Evaluation with metrics
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Data loading and label encoding
│   │   └── dataset.py          # PyTorch datasets
│   ├── models/
│   │   ├── baseline.py         # Simple classifiers
│   │   └── multitask.py        # Multi-task architectures
│   └── utils/
│       ├── losses.py           # Focal loss, weighted BCE
│       └── sequence_handling.py # Long sequence truncation/chunking
│
├── eda.py                       # Basic data exploration
├── eda_advanced.py              # Advanced EDA with visualizations
│
└── cafa-6-protein-function-prediction/  # Dataset (not in git)
    ├── Train/
    │   ├── train_sequences.fasta
    │   ├── train_terms.tsv
    │   └── go-basic.obo
    └── Test/
        └── testsuperset.fasta
```

---

## Model Architecture

### Baseline Approach (Recommended for First Run)
```
Protein Sequence → ESM-2 (pre-trained) → [CLS] Embedding (640-dim)
                                              ↓
                                         Dense(640 → 512)
                                              ↓
                                         ReLU + Dropout(0.3)
                                              ↓
                                         Dense(512 → 26,125)
                                              ↓
                                         Sigmoid → Predictions
```

### Multi-Task Architecture (Better Performance)
```
                                        ┌→ C Head (512 → 2,651) → C Predictions
                                        │
Embedding (640-dim) → Shared(640 → 1024) ─→ F Head (512 → 6,616) → F Predictions
                                        │
                                        └→ P Head (512 → 16,858) → P Predictions
```

**Loss**: Focal loss with learnable task weights (handles extreme imbalance)

---

## Key Features

- **ESM-2 Embeddings**: State-of-the-art protein language model
- **Multi-Task Learning**: Separate heads for C/F/P aspects
- **Focal Loss**: Handles extreme class imbalance (51.6% rare labels)
- **Balanced Truncation**: Preserves N-term + C-term for long sequences
- **GPU Optimization**: Dual-GPU support for faster embedding generation
- **Kaggle-Ready**: Memory-efficient scripts for Kaggle notebooks
- **Threshold Optimization**: Per-label threshold tuning (+3-5% F1)

---

## Performance Expectations

| Model | Embeddings | Training F1 | Val F1 | Time |
|-------|-----------|-------------|--------|------|
| Baseline | ESM-2 150M | 0.50-0.60 | 0.45-0.55 | ~2 hours |
| Multi-Task | ESM-2 150M | 0.55-0.65 | 0.50-0.60 | ~2.5 hours |
| Multi-Task | ESM-2 650M | 0.60-0.70 | 0.55-0.65 | ~4 hours |

**Note**: Times include embedding generation + training on Kaggle P100 GPU

---

## Documentation

### Analysis & Decisions
**[CLAUDE.md](CLAUDE.md)** - EDA findings and implementation rationale:
- Dataset analysis and challenges
- Model selection reasoning
- Architecture decisions
- Expected performance
- Lessons learned

### For Kaggle Users
**[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)** - Complete step-by-step workflow:
- Setup and dataset upload
- Embedding generation (P100 vs T4 x2)
- Training and evaluation
- Submission creation and upload
- Troubleshooting

### For Technical Deep-Dive
**[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)** - Advanced topics:
- ESM-2 model comparison (8M/150M/650M/3B)
- Long sequence handling strategies
- GPU optimization (P100 vs T4 x2)
- Multi-task learning details
- Loss function comparison

---

## Kaggle Workflow (TL;DR)

```python
# 1. Generate embeddings (~1.5 hours)
from generate_embeddings_kaggle import generate_embeddings_kaggle
embeddings = generate_embeddings_kaggle(
    data_dir='/kaggle/input/cafa-6-protein-function-prediction',
    model_name='esm2_t30_150M_UR50D'
)

# 2. Train model (~30 minutes)
!python scripts/train_baseline.py --embeddings {embeddings}

# 3. Make predictions (~5 minutes)
!python scripts/evaluate.py --model models/baseline/best_model.pt

# 4. Submit to competition
# See KAGGLE_GUIDE.md for submission format
```

**Total time**: ~2-3 hours from start to first submission

---

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 2

# Or use smaller model
--model_name esm2_t30_150M_UR50D
```

### Only Using 1 GPU on T4 x2
```python
# Use dual-GPU script
from generate_embeddings_kaggle_dual_gpu import generate_embeddings_dual_gpu
```

### ImportError: fair-esm
```bash
pip install fair-esm
# or
pip install git+https://github.com/facebookresearch/esm.git
```

---

## Improvements Roadmap

Baseline is competitive (~0.45-0.55 F1). To improve further:

1. **Threshold Optimization** (+3-5% F1) - Use `--optimize_threshold`
2. **Larger Model** (+5-8% F1) - Switch to ESM-2 650M
3. **Sliding Window** (+2-3% F1) - For long sequences (see TECHNICAL_NOTES.md)
4. **Ensemble** (+3-5% F1) - Average multiple models
5. **Multi-Task Learning** (+5-10% F1) - Use separate C/F/P heads

**Target**: 0.55-0.60 F1 with all optimizations (competitive for top 20%)

---

## Resources

- **Competition**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- **ESM-2 Paper**: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1
- **ESM GitHub**: https://github.com/facebookresearch/esm
- **Gene Ontology**: http://geneontology.org/

---

## License

This project is for educational and competition purposes.

---

**Quick Links**:
- [Analysis & Decisions](CLAUDE.md) - EDA findings and implementation rationale
- [Kaggle Workflow](KAGGLE_GUIDE.md) - Step-by-step guide for Kaggle notebooks
- [Technical Details](TECHNICAL_NOTES.md) - Model architecture, GPU optimization, advanced topics
- [Competition Page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
