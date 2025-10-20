# Quick Start Guide: CAFA-6 Baseline

Get your first baseline model running in **2-3 hours** using ESM-2 embeddings.

## Prerequisites

```bash
# Install dependencies
pip install torch fair-esm pandas numpy scikit-learn tqdm
```

## Model Selection

**Recommended for quick start:** ESM-2 150M (`esm2_t30_150M_UR50D`)
- ⚡ 2x faster than 650M model (2 hours vs 3 hours)
- 💾 Half the storage (21 GB vs 42 GB)
- 📊 90% performance of largest model
- ✅ Perfect for baseline and iteration

See `ESM2_MODEL_COMPARISON.md` for detailed comparison of all models.

## Step 1: Generate ESM-2 Embeddings (~2 hours with 150M model)

### Option A: Local GPU

```bash
# Navigate to project directory
cd cafa6

# Generate embeddings for training set
python scripts/generate_embeddings.py \
    --data_dir cafa-6-protein-function-prediction \
    --output_dir data/embeddings \
    --split train \
    --model_name esm2_t30_150M_UR50D \
    --strategy balanced \
    --batch_size 8 \
    --device cuda

# Alternative: Use faster 8M model for quick testing (30 min)
# --model_name esm2_t6_8M_UR50D --batch_size 16

# Alternative: Use 650M model for best performance (3 hours)
# --model_name esm2_t33_650M_UR50D --batch_size 8
```

**Expected output:**
```
Loading protein sequences...
Loaded 82404 train sequences

Initializing ESM-2 model: esm2_t30_150M_UR50D
Device: cuda
Strategy: balanced
Max length: 1024
Embedding dimension: 640

Generating embeddings for 82404 proteins...
Batch size: 8

Generating embeddings: 100%|██████████| 10301/10301 [1:52:15<00:00,  1.52it/s]

Embedding generation complete!
Total proteins: 82404
Truncated sequences: 20456 (24.8%)
Average AA lost per truncated sequence: 487.3
Embedding shape: (82404, 640)

Embeddings saved to: data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl
File size: 21.17 GB
```

### Option B: Kaggle Notebook

```python
# In Kaggle notebook cell 1
!pip install fair-esm

# In cell 2
from generate_embeddings_kaggle import generate_embeddings_kaggle

embeddings_path = generate_embeddings_kaggle(
    data_dir='/kaggle/input/cafa-6-protein-function-prediction',
    output_dir='/kaggle/working/embeddings',
    model_name='esm2_t33_650M_UR50D',
    split='train',
    batch_size=4,
    strategy='balanced'
)
```

## Step 2: Verify Embeddings (~1 minute)

```bash
python scripts/test_embeddings.py \
    data/embeddings/train_embeddings_esm2_t33_650M_UR50D_balanced.pkl
```

Should see:
```
✓ ALL TESTS PASSED!
```

## Step 3: Preprocess Labels (~5 minutes)

```bash
python -c "
from src.data import ProteinDataPreprocessor

preprocessor = ProteinDataPreprocessor('cafa-6-protein-function-prediction')
data = preprocessor.preprocess_all(output_dir='data/processed')
"
```

**Expected output:**
```
Total proteins: 82404
Total GO terms: 26125
Total annotations: 537027
Aspects: C, F, P
Preprocessed data saved to data/processed/preprocessed_data.pkl
```

## Step 4: Train Baseline Model (~30 minutes)

Create a simple training script `train_simple.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np

# Load embeddings
with open('data/embeddings/train_embeddings_esm2_t33_650M_UR50D_balanced.pkl', 'rb') as f:
    emb_data = pickle.load(f)

# Load labels
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    label_data = pickle.load(f)

# Quick dataset
from src.data import ProteinEmbeddingDataset

dataset = ProteinEmbeddingDataset(
    protein_ids=label_data['protein_ids'],
    embeddings=emb_data['embeddings'],
    labels=label_data['full_labels']
)

# Train/val split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Simple model
from src.models import SimpleProteinClassifier
from src.utils import FocalLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SimpleProteinClassifier(
    embedding_dim=1280,
    num_classes=26125,
    hidden_dims=[1024, 512]
).to(device)

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
print("Training baseline model...")
for epoch in range(5):  # Quick baseline: just 5 epochs
    model.train()
    total_loss = 0

    for batch in train_loader:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'models/baseline_model.pt')
print("Model saved to models/baseline_model.pt")
```

Run it:
```bash
python train_simple.py
```

**Expected output:**
```
Training baseline model...
Epoch 1/5, Loss: 0.0234
Epoch 2/5, Loss: 0.0187
Epoch 3/5, Loss: 0.0156
Epoch 4/5, Loss: 0.0142
Epoch 5/5, Loss: 0.0135
Model saved to models/baseline_model.pt
```

## Step 5: Evaluate (~5 minutes)

```python
# Evaluate on validation set
model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for batch in val_loader:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels']

        outputs = model(embeddings)
        probs = torch.sigmoid(outputs).cpu()

        predictions.append(probs)
        ground_truth.append(labels)

predictions = torch.cat(predictions, dim=0).numpy()
ground_truth = torch.cat(ground_truth, dim=0).numpy()

# Calculate metrics
from sklearn.metrics import f1_score

# Use threshold of 0.5
pred_binary = (predictions > 0.5).astype(int)

f1_micro = f1_score(ground_truth, pred_binary, average='micro')
f1_macro = f1_score(ground_truth, pred_binary, average='macro', zero_division=0)

print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
```

**Expected scores (ballpark):**
```
F1 Micro: 0.45-0.55
F1 Macro: 0.15-0.25
```

## What You Just Did

1. ✅ Generated ESM-2 embeddings (1280-dim vectors for each protein)
2. ✅ Preprocessed GO term labels into binary matrices
3. ✅ Trained a simple 3-layer neural network
4. ✅ Evaluated with F1 scores

**This is your baseline!** Now you can iterate:

## Next Improvements (In Order)

### Improvement 1: Multi-Task Learning

Replace single-task model with multi-task (separate heads for C/F/P):

```python
from src.models import MultiTaskProteinClassifier
from src.utils import MultiTaskLoss

model = MultiTaskProteinClassifier(
    embedding_dim=1280,
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858
)

criterion = MultiTaskLoss(
    num_classes_C=2651,
    num_classes_F=6616,
    num_classes_P=16858,
    loss_type='focal',
    learnable_weights=True
)
```

**Expected improvement:** +5-10% F1

### Improvement 2: Threshold Optimization

Instead of fixed 0.5 threshold, optimize per-label:

```python
from sklearn.metrics import f1_score

best_thresholds = []
for label_idx in range(26125):
    best_thresh = 0.5
    best_f1 = 0

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (predictions[:, label_idx] > thresh).astype(int)
        f1 = f1_score(ground_truth[:, label_idx], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    best_thresholds.append(best_thresh)
```

**Expected improvement:** +3-5% F1

### Improvement 3: Class Weights

Use class weights to handle imbalance:

```python
# Load class weights from preprocessing
class_weights = label_data['class_weights']

criterion = FocalLoss(alpha=0.25, gamma=2.0)
# Or
from src.utils import WeightedBCELoss
criterion = WeightedBCELoss(class_weights=torch.FloatTensor(class_weights))
```

**Expected improvement:** +2-5% F1 on rare classes

### Improvement 4: Fine-tune ESM-2

Instead of pre-computed embeddings, fine-tune the encoder:

```python
from src.models import ESM2Classifier

model = ESM2Classifier(
    esm_model_name='esm2_t33_650M_UR50D',
    num_classes=26125,
    freeze_encoder=True,  # Start frozen
    num_unfrozen_layers=2  # Then unfreeze top 2 layers
)
```

**Expected improvement:** +10-15% F1 (but slower training)

## Timeline Summary

| Step | Time | GPU/CPU | Output |
|------|------|---------|--------|
| Generate embeddings | 3 hours | GPU | 42 GB file |
| Preprocess labels | 5 min | CPU | 500 MB file |
| Train baseline | 30 min | GPU | F1 ~0.50 |
| Multi-task | +1 hour | GPU | F1 ~0.55 |
| Threshold opt | +30 min | CPU | F1 ~0.58 |
| Fine-tune ESM-2 | +6 hours | GPU | F1 ~0.65+ |

**Total to competitive model:** ~12 hours of GPU time

## Troubleshooting

### "CUDA out of memory"

```python
# Reduce batch size
batch_size=16  # or 8, or 4
```

### "Embeddings file too large"

```python
# Use smaller ESM-2 model
model_name='esm2_t30_150M_UR50D'  # 640-dim instead of 1280-dim
```

### "Training is slow"

```python
# Use more workers for data loading
train_loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

### "F1 score is very low (< 0.3)"

Check:
- Are you using sigmoid activation? `torch.sigmoid(outputs)`
- Is the loss decreasing? Should go from ~0.03 to ~0.01
- Are labels correct? `labels.sum()` should be ~6.5 per protein

## Next Steps

1. **Read CLAUDE.md** for detailed analysis
2. **Read MODEL_IMPLEMENTATION.md** for architecture details
3. **Read LONG_SEQUENCE_GUIDE.md** for handling long proteins
4. **Join Kaggle discussion** to see what others are doing

## Questions?

Check the documentation:
- `README.md` - Project overview
- `CLAUDE.md` - Detailed EDA and recommendations
- `MODEL_IMPLEMENTATION.md` - Code structure
- `scripts/README.md` - Script documentation

Good luck! 🚀
