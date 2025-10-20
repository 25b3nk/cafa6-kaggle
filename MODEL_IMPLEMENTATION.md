# Model Implementation Summary

## What We've Built

This document summarizes the data preprocessing and model architecture code that's been implemented.

---

## Project Structure

```
cafa6/
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Data preprocessing utilities
│   │   └── dataset.py             # PyTorch dataset classes
│   ├── models/
│   │   ├── baseline.py            # Baseline model architectures
│   │   └── multitask.py           # Multi-task model architectures
│   └── utils/
│       └── losses.py              # Loss functions (Focal, Asymmetric, etc.)
├── notebooks/                     # For Kaggle notebooks (TBD)
├── eda.py                         # Basic EDA script
├── eda_advanced.py                # Advanced EDA script
├── CLAUDE.md                      # Detailed analysis and recommendations
└── README.md                      # Project overview
```

---

## 1. Data Preprocessing (`src/data/preprocessing.py`)

### Features:
- **ProteinDataPreprocessor** class for complete data pipeline
- Loads FASTA sequences and TSV annotations
- Creates GO term to index mappings
- Builds multi-label binary matrices
- Computes class weights for imbalance handling
- Supports both full and aspect-specific (C/F/P) label matrices
- Saves preprocessed data to pickle for fast loading

### Key Functions:
```python
preprocessor = ProteinDataPreprocessor('cafa-6-protein-function-prediction')
data = preprocessor.preprocess_all(output_dir='data/processed')
```

---

## 2. PyTorch Datasets (`src/data/dataset.py`)

### Dataset Classes:

1. **ProteinSequenceDataset**
   - For raw protein sequences
   - Supports tokenization (ESM, ProtBERT)
   - Handles variable sequence lengths

2. **ProteinEmbeddingDataset**
   - For pre-computed embeddings (faster)
   - Good for baseline experiments

3. **MultiAspectProteinDataset**
   - Separate labels for C, F, P aspects
   - For multi-task learning

4. **MultiAspectEmbeddingDataset**
   - Pre-computed embeddings + multi-aspect labels
   - Best for quick multi-task experiments

### Usage:
```python
train_loader, val_loader = create_dataloaders(
    'data/processed/preprocessed_data.pkl',
    batch_size=32,
    multi_aspect=True,  # For multi-task learning
    tokenizer=None      # Add tokenizer if using sequences
)
```

---

## 3. Baseline Models (`src/models/baseline.py`)

### Model Architectures:

1. **SimpleProteinClassifier**
   - Dense NN on pre-computed embeddings
   - Fast baseline for experimentation
   - 2-3 hidden layers with dropout

2. **ESM2Classifier**
   - Uses ESM-2 protein language model
   - Can freeze or fine-tune encoder
   - State-of-the-art protein understanding

3. **ProtBERTClassifier**
   - Uses ProtBERT from HuggingFace
   - Alternative to ESM-2

4. **CNNProteinClassifier**
   - 1D convolutions for motif detection
   - Multiple kernel sizes
   - Good for finding local patterns

### Usage:
```python
# Simple baseline
model = SimpleProteinClassifier(
    embedding_dim=1280,
    num_classes=26125,
    hidden_dims=[1024, 512]
)

# ESM-2 model
model = ESM2Classifier(
    esm_model_name='esm2_t33_650M_UR50D',
    num_classes=26125,
    freeze_encoder=True  # Start frozen, fine-tune later
)
```

---

## 4. Multi-Task Models (`src/models/multitask.py`)

### Model Architectures:

1. **MultiTaskProteinClassifier**
   - Shared encoder + 3 separate heads (C/F/P)
   - Uses pre-computed embeddings
   - Fast multi-task baseline

2. **MultiTaskESM2Classifier**
   - ESM-2 encoder + 3 aspect heads
   - Recommended for best performance
   - Can fine-tune encoder layers

3. **HierarchicalMultiTaskClassifier**
   - Predicts aspect presence first
   - Then predicts GO terms per aspect
   - Auxiliary task for better learning

### Usage:
```python
model = MultiTaskESM2Classifier(
    esm_model_name='esm2_t33_650M_UR50D',
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858,
    freeze_encoder=True,
    num_unfrozen_layers=2  # Fine-tune top 2 layers
)

# Forward pass
outputs = model(input_ids, attention_mask)
# Returns: {'C': logits_C, 'F': logits_F, 'P': logits_P}
```

---

## 5. Loss Functions (`src/utils/losses.py`)

### Available Losses:

1. **FocalLoss**
   - Focuses on hard-to-learn examples
   - Down-weights easy examples
   - Good for extreme imbalance

2. **WeightedBCELoss**
   - Standard BCE with class weights
   - Simple and effective

3. **AsymmetricLoss**
   - Different focusing for positive/negative
   - State-of-the-art for multi-label

4. **MultiTaskLoss**
   - Combines C/F/P losses
   - Learnable task weights (uncertainty weighting)
   - For multi-task models

### Usage:
```python
# For single-task models
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, labels)

# For multi-task models
criterion = MultiTaskLoss(
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858,
    loss_type='focal',
    learnable_weights=True  # Learn task importance
)

outputs = {'C': logits_C, 'F': logits_F, 'P': logits_P}
targets = {'C': labels_C, 'F': labels_F, 'P': labels_P}
total_loss, loss_dict = criterion(outputs, targets)
```

---

## Recommended Workflow

### Stage 1: Quick Baseline (Local Development)

```python
# 1. Preprocess data
from src.data import ProteinDataPreprocessor
preprocessor = ProteinDataPreprocessor('cafa-6-protein-function-prediction')
data = preprocessor.preprocess_all(output_dir='data/processed')

# 2. Create simple model (assuming you have embeddings)
from src.models import SimpleProteinClassifier
model = SimpleProteinClassifier(
    embedding_dim=1280,
    num_classes=26125
)

# 3. Use focal loss
from src.utils import FocalLoss
criterion = FocalLoss()

# 4. Train and evaluate
# (Training loop to be implemented)
```

### Stage 2: Multi-Task Model (Kaggle GPU)

```python
# 1. Load preprocessed data
from src.data import create_dataloaders
train_loader, val_loader = create_dataloaders(
    'preprocessed_data.pkl',
    batch_size=32,
    multi_aspect=True
)

# 2. Create multi-task model
from src.models import MultiTaskESM2Classifier
model = MultiTaskESM2Classifier(
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858,
    freeze_encoder=True
)

# 3. Use multi-task loss
from src.utils import MultiTaskLoss
criterion = MultiTaskLoss(
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858,
    loss_type='focal',
    learnable_weights=True
)

# 4. Train on Kaggle GPU
# (Training loop to be implemented)
```

---

## Next Steps

### Immediate (This Week):
1. ✅ Data preprocessing - **DONE**
2. ✅ Dataset classes - **DONE**
3. ✅ Model architectures - **DONE**
4. ✅ Loss functions - **DONE**
5. ⏳ Training loop implementation
6. ⏳ Evaluation metrics (F1, precision, recall)
7. ⏳ Create Kaggle notebook

### Short Term (Next Week):
1. Train baseline model
2. Implement threshold optimization
3. Add validation metrics
4. Create submission pipeline
5. Fine-tune ESM-2 encoder

### Long Term (As Needed):
1. Ensemble multiple models
2. Pseudo-labeling on test set
3. GO hierarchy constraints
4. Advanced augmentation

---

## Key Design Decisions

1. **Modular Structure**: Easy to swap models, losses, datasets
2. **Multi-Task by Default**: C/F/P aspects handled separately
3. **Flexibility**: Supports both embeddings and raw sequences
4. **Imbalance Handling**: Focal loss + class weights
5. **Kaggle-Ready**: Designed for Kaggle GPU constraints

---

## Dependencies

```bash
# Core
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn

# Protein models
pip install fair-esm  # ESM-2
pip install transformers  # ProtBERT

# Utils
pip install biopython  # Sequence handling
pip install matplotlib seaborn  # Visualization
```

---

## Notes

- All code is tested locally but needs GPU for full training
- ESM-2 `esm2_t33_650M_UR50D` is ~2.5GB - fits in Kaggle GPU memory
- Preprocessing can be done locally, training on Kaggle
- Multi-task model is recommended based on EDA findings

---

**Status**: Data preprocessing and model architectures are complete. Ready to implement training loop and create Kaggle notebook.
