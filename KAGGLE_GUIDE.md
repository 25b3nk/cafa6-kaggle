# Running CAFA-6 Baseline on Kaggle - Complete Guide

This guide shows you how to run the complete baseline pipeline on Kaggle notebooks using free GPU.

**Other Documentation**:
- [README.md](README.md) - Project overview and quick start
- [CLAUDE.md](CLAUDE.md) - EDA findings and implementation rationale
- [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) - Model details, GPU optimization, advanced topics

---

## Prerequisites

1. **Kaggle account** (free tier is fine)
2. **Competition data** already available on Kaggle
3. **GPU accelerator** enabled in notebook settings

---

## Option 1: Quick Start (Single Notebook)

### Step 1: Create New Kaggle Notebook

1. Go to https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/code
2. Click **"New Notebook"**
3. Enable **GPU** in settings (right panel → Accelerator → GPU T4 x2)
4. Enable **Internet** (to install packages)

### Step 2: Upload Your Code as Dataset

**Option A: Upload as ZIP**
1. On your local machine, create a ZIP of the `src/` directory:
   ```bash
   cd cafa6
   zip -r cafa6-code.zip src/
   ```

2. On Kaggle:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `cafa6-code.zip`
   - Name it "cafa6-baseline-code"
   - Make it private
   - Create

3. Add this dataset to your notebook:
   - In notebook, click "Add Data" → "Your Datasets" → "cafa6-baseline-code"

**Option B: Copy-paste code directly (simpler for quick start)**
- We'll do this method in the notebook cells below

### Step 3: Notebook Cell 1 - Setup

```python
# Cell 1: Install dependencies
!pip install fair-esm -q

import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 4: Notebook Cell 2 - Create Source Code

```python
# Cell 2: Create necessary directories and copy minimal required code

!mkdir -p src/data src/models src/utils

# We'll create minimal versions of required files inline
# (Alternative: upload as dataset and unzip)

print("✓ Directories created")
print("Next: Copy minimal code for preprocessing, models, and training")
```

### Step 5: Notebook Cell 3 - Preprocessing Code

```python
# Cell 3: Create minimal preprocessing utilities

preprocessing_code = '''
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class ProteinDataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "Train"
        self.go_term_to_idx = {}
        self.idx_to_go_term = {}
        self.go_term_to_aspect = {}
        self.aspect_to_terms = {"C": [], "F": [], "P": []}

    def load_sequences(self, split="train"):
        if split == "train":
            fasta_file = self.train_dir / "train_sequences.fasta"
        else:
            fasta_file = self.data_dir / "Test" / "testsuperset.fasta"

        sequences = {}
        current_id = None
        current_seq = []

        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]
                    if "|" in current_id:
                        current_id = current_id.split("|")[1]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                sequences[current_id] = "".join(current_seq)

        print(f"Loaded {len(sequences)} {split} sequences")
        return sequences

    def load_annotations(self):
        annotations_file = self.train_dir / "train_terms.tsv"
        df = pd.read_csv(annotations_file, sep="\\t")
        print(f"Loaded {len(df)} annotations")
        return df

    def build_label_encodings(self, annotations_df):
        unique_terms = sorted(annotations_df["term"].unique())
        self.go_term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
        self.idx_to_go_term = {idx: term for term, idx in self.go_term_to_idx.items()}

        term_aspect_map = annotations_df[["term", "aspect"]].drop_duplicates()
        self.go_term_to_aspect = dict(zip(term_aspect_map["term"], term_aspect_map["aspect"]))

        for term, aspect in self.go_term_to_aspect.items():
            self.aspect_to_terms[aspect].append(term)

        print(f"Total GO terms: {len(self.go_term_to_idx)}")

    def create_multi_label_matrix(self, annotations_df, protein_ids):
        n_proteins = len(protein_ids)
        n_terms = len(self.go_term_to_idx)
        labels = np.zeros((n_proteins, n_terms), dtype=np.float32)

        protein_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}

        for _, row in annotations_df.iterrows():
            protein_id = row["EntryID"]
            term = row["term"]

            if protein_id in protein_to_idx and term in self.go_term_to_idx:
                protein_idx = protein_to_idx[protein_id]
                term_idx = self.go_term_to_idx[term]
                labels[protein_idx, term_idx] = 1.0

        print(f"Created label matrix: {labels.shape}")
        return labels

    def create_aspect_specific_matrices(self, annotations_df, protein_ids):
        aspect_data = {}

        for aspect in ["C", "F", "P"]:
            aspect_terms = self.aspect_to_terms[aspect]
            aspect_term_indices = [self.go_term_to_idx[term] for term in aspect_terms]
            aspect_idx_map = {global_idx: local_idx
                            for local_idx, global_idx in enumerate(aspect_term_indices)}

            n_proteins = len(protein_ids)
            n_aspect_terms = len(aspect_term_indices)
            labels = np.zeros((n_proteins, n_aspect_terms), dtype=np.float32)

            protein_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
            aspect_annotations = annotations_df[annotations_df["aspect"] == aspect]

            for _, row in aspect_annotations.iterrows():
                protein_id = row["EntryID"]
                term = row["term"]

                if protein_id in protein_to_idx and term in self.go_term_to_idx:
                    protein_idx = protein_to_idx[protein_id]
                    global_term_idx = self.go_term_to_idx[term]
                    local_term_idx = aspect_idx_map[global_term_idx]
                    labels[protein_idx, local_term_idx] = 1.0

            aspect_data[aspect] = (labels, aspect_term_indices)
            print(f"Aspect {aspect}: {labels.shape}")

        return aspect_data
'''

with open('/kaggle/working/preprocessing.py', 'w') as f:
    f.write(preprocessing_code)

print("✓ Preprocessing code created")
```

### Step 6: Notebook Cell 4 - Generate ESM-2 Embeddings

```python
# Cell 4: Generate embeddings (THIS IS THE LONGEST STEP - ~2 hours)

import torch
import numpy as np
from tqdm import tqdm
import esm
import pickle
import gc

# Configuration
MODEL_NAME = 'esm2_t30_150M_UR50D'  # Using 150M for speed
BATCH_SIZE = 4  # Small batch for Kaggle GPU
MAX_LENGTH = 1024
DATA_DIR = '/kaggle/input/cafa-6-protein-function-prediction'

print(f"Loading ESM-2 model: {MODEL_NAME}")
model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
batch_converter = alphabet.get_batch_converter()
model = model.cuda()
model.eval()

# Load sequences
sys.path.append('/kaggle/working')
from preprocessing import ProteinDataPreprocessor

preprocessor = ProteinDataPreprocessor(DATA_DIR)
sequences = preprocessor.load_sequences('train')
protein_ids = list(sequences.keys())

print(f"\nGenerating embeddings for {len(protein_ids)} proteins...")

# Truncation helper
def truncate_balanced(seq, max_len):
    if len(seq) <= max_len:
        return seq
    n_term = max_len // 2
    c_term = max_len - n_term
    return seq[:n_term] + seq[-c_term:]

# Generate embeddings in batches
all_embeddings = []
all_protein_ids = []

for i in tqdm(range(0, len(protein_ids), BATCH_SIZE)):
    batch_ids = protein_ids[i:i + BATCH_SIZE]
    batch_data = [(pid, truncate_balanced(sequences[pid], MAX_LENGTH))
                  for pid in batch_ids]

    # Convert and embed
    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=False)
        embeddings = results['representations'][30][:, 0, :].cpu().numpy()

    all_embeddings.extend(embeddings)
    all_protein_ids.extend(batch_ids)

    # Clear GPU memory periodically
    if (i // BATCH_SIZE) % 100 == 0:
        torch.cuda.empty_cache()
        gc.collect()

embeddings_array = np.array(all_embeddings)

# Save embeddings
embeddings_data = {
    'embeddings': embeddings_array,
    'protein_ids': all_protein_ids,
    'model_name': MODEL_NAME,
    'embedding_dim': model.embed_dim,
    'num_proteins': len(all_protein_ids)
}

with open('/kaggle/working/train_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_data, f)

print(f"\n✓ Embeddings generated: {embeddings_array.shape}")
print(f"✓ Saved to /kaggle/working/train_embeddings.pkl")
print(f"Size: {Path('/kaggle/working/train_embeddings.pkl').stat().st_size / 1e9:.2f} GB")

# Free GPU memory
del model
torch.cuda.empty_cache()
gc.collect()
```

**⏰ This cell will take ~2 hours. You can:**
- Go grab coffee ☕
- Work on something else
- Monitor GPU usage: `!nvidia-smi` in a new cell

### Step 7: Notebook Cell 5 - Preprocess Labels

```python
# Cell 5: Preprocess labels (~5 minutes)

sys.path.append('/kaggle/working')
from preprocessing import ProteinDataPreprocessor

preprocessor = ProteinDataPreprocessor(DATA_DIR)
sequences = preprocessor.load_sequences('train')
annotations_df = preprocessor.load_annotations()
preprocessor.build_label_encodings(annotations_df)

protein_ids = sorted(list(set(sequences.keys()) & set(annotations_df['EntryID'].unique())))
print(f"Proteins with both sequence and annotations: {len(protein_ids)}")

# Create label matrices
full_labels = preprocessor.create_multi_label_matrix(annotations_df, protein_ids)
aspect_labels = preprocessor.create_aspect_specific_matrices(annotations_df, protein_ids)

# Save
preprocessed_data = {
    'protein_ids': protein_ids,
    'full_labels': full_labels,
    'aspect_labels': aspect_labels,
    'go_term_to_idx': preprocessor.go_term_to_idx,
    'idx_to_go_term': preprocessor.idx_to_go_term,
    'go_term_to_aspect': preprocessor.go_term_to_aspect
}

with open('/kaggle/working/preprocessed_labels.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("✓ Labels preprocessed and saved")
```

### Step 8: Notebook Cell 6 - Simple Training (Inline)

```python
# Cell 6: Train simple baseline model (~30 minutes)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Simple dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'labels': self.labels[idx]
        }

# Focal Loss (simplified)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

# Load data
with open('/kaggle/working/train_embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
with open('/kaggle/working/preprocessed_labels.pkl', 'rb') as f:
    label_data = pickle.load(f)

embeddings = emb_data['embeddings']
labels = label_data['full_labels']

print(f"Embeddings: {embeddings.shape}")
print(f"Labels: {labels.shape}")

# Create dataset
dataset = EmbeddingDataset(embeddings, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = SimpleClassifier(
    input_dim=emb_data['embedding_dim'],
    num_classes=labels.shape[1]
).cuda()

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"\n{'='*60}")
print("Training baseline model...")
print(f"{'='*60}\n")

# Training loop
for epoch in range(10):
    # Train
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        embeddings = batch['embedding'].cuda()
        labels_batch = batch['labels'].cuda()

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embedding'].cuda()
            labels_batch = batch['labels'].cuda()

            outputs = model(embeddings)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_batch.cpu().numpy())

    # Calculate F1
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int),
                  average='micro', zero_division=0)

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Loss={val_loss/len(val_loader):.4f}, F1={f1:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_dim': emb_data['embedding_dim'],
    'num_classes': labels.shape[1]
}, '/kaggle/working/baseline_model.pt')

print(f"\n✓ Model trained and saved!")
print(f"Final F1: {f1:.4f}")
```

### Step 9: Results

After completing all cells, you should have:

```
/kaggle/working/
├── train_embeddings.pkl      # ~21 GB (150M model)
├── preprocessed_labels.pkl   # ~500 MB
├── baseline_model.pt         # ~20 MB
└── preprocessing.py          # Helper code
```

**Expected F1 Score:** 0.50-0.60

---

## Option 2: Using Uploaded Code as Dataset

If you upload your `src/` code as a dataset, the notebook is much simpler:

### Notebook with Code Dataset:

```python
# Cell 1: Setup
!pip install fair-esm -q

import sys
sys.path.append('/kaggle/input/cafa6-baseline-code')  # Your uploaded code

from src.data import ProteinDataPreprocessor
from src.models import SimpleProteinClassifier
from src.utils import FocalLoss

# Cell 2: Run embedding generation script
# (Copy generate_embeddings_kaggle.py content here)

# Cell 3: Train using train_baseline.py logic
# Much cleaner!
```

---

## Time Breakdown on Kaggle

| Step | Time | Description |
|------|------|-------------|
| Cell 1-3 | 5 min | Setup and code creation |
| Cell 4 | **2 hours** | ESM-2 embedding generation ⏰ |
| Cell 5 | 5 min | Label preprocessing |
| Cell 6 | 30 min | Model training |
| **Total** | **~2.5 hours** | Complete baseline |

---

## Tips for Kaggle

### Memory Management
```python
# Clear GPU memory between steps
import gc
import torch

del model  # Delete large objects
torch.cuda.empty_cache()
gc.collect()
```

### Check GPU Usage
```python
# Monitor GPU
!nvidia-smi

# Check available memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Save Intermediate Results
```python
# Always save after expensive steps
# If notebook crashes, you won't lose embeddings!

# After embeddings
pickle.dump(embeddings_data, open('/kaggle/working/embeddings.pkl', 'wb'))

# After training
torch.save(model.state_dict(), '/kaggle/working/model.pt')
```

### Download Results
```python
# Download trained model to your computer
from IPython.display import FileLink
FileLink('/kaggle/working/baseline_model.pt')
```

---

## Troubleshooting on Kaggle

### "CUDA Out of Memory"
```python
# Reduce batch size
BATCH_SIZE = 2  # Instead of 4 or 8
```

### "Notebook Timeout (9 hours)"
```python
# Save embeddings as Kaggle dataset after generation
# Then create new notebook that loads them
# This way embedding generation doesn't count toward 9h limit
```

### "Too Large to Save"
```python
# Compress embeddings
embeddings_compressed = embeddings.astype(np.float16)  # Half precision
# Saves 50% space, minimal accuracy loss
```

---

## Step 10: Generate Test Embeddings

```python
# Cell 7: Generate embeddings for test set (~30 minutes)

import torch
import esm
import pickle
import gc
from tqdm import tqdm

print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
batch_converter = alphabet.get_batch_converter()
model = model.cuda()
model.eval()

# Load test sequences
print("Loading test sequences...")
sys.path.append('/kaggle/working')
from preprocessing import ProteinDataPreprocessor

preprocessor = ProteinDataPreprocessor(DATA_DIR)
test_sequences = preprocessor.load_sequences('test')
test_protein_ids = list(test_sequences.keys())

print(f"Generating embeddings for {len(test_protein_ids)} test proteins...")

# Generate embeddings
test_embeddings = []
test_ids = []

for i in tqdm(range(0, len(test_protein_ids), BATCH_SIZE)):
    batch_ids = test_protein_ids[i:i + BATCH_SIZE]
    batch_data = [(pid, truncate_balanced(test_sequences[pid], MAX_LENGTH))
                  for pid in batch_ids]

    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=False)
        embeddings = results['representations'][30][:, 0, :].cpu().numpy()

    test_embeddings.extend(embeddings)
    test_ids.extend(batch_ids)

    if (i // BATCH_SIZE) % 100 == 0:
        torch.cuda.empty_cache()
        gc.collect()

test_embeddings_array = np.array(test_embeddings)

# Save test embeddings
test_emb_data = {
    'embeddings': test_embeddings_array,
    'protein_ids': test_ids,
    'model_name': MODEL_NAME,
    'embedding_dim': model.embed_dim
}

with open('/kaggle/working/test_embeddings.pkl', 'wb') as f:
    pickle.dump(test_emb_data, f)

print(f"\n✓ Test embeddings generated: {test_embeddings_array.shape}")

# Free memory
del model
torch.cuda.empty_cache()
gc.collect()
```

---

## Step 11: Create Submission File (Memory-Efficient)

**IMPORTANT**: The old approach loads all predictions into RAM and causes OOM errors. Use this memory-efficient version instead!

```python
# Cell 8: Create submission file (Memory-Efficient, TSV format)
# This avoids RAM overload by streaming predictions directly to file

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc

print("Loading data...")

# Load test embeddings
with open('/kaggle/working/test_embeddings.pkl', 'rb') as f:
    test_data = pickle.load(f)
test_embeddings = test_data['embeddings']
protein_ids = test_data['protein_ids']

# Load model
checkpoint = torch.load('/kaggle/working/baseline_model.pt')
model = SimpleClassifier(
    input_dim=checkpoint['embedding_dim'],
    num_classes=checkpoint['num_classes']
).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load GO term mappings
with open('/kaggle/working/preprocessed_labels.pkl', 'rb') as f:
    label_data = pickle.load(f)
idx_to_go_term = label_data['idx_to_go_term']

print(f"Generating predictions for {len(protein_ids)} proteins...")

# ============================================================
# Memory-Efficient Submission Creation (TSV format)
# ============================================================

output_file = '/kaggle/working/submission.tsv'
threshold = 0.5  # Adjust if needed
batch_size = 16  # Small batch to avoid OOM

# Write header (TSV format - tab separated)
with open(output_file, 'w') as f:
    f.write("Protein ID\tGO Term\tConfidence\n")

total_preds = 0
no_pred_count = 0

# Process in batches and write immediately
with torch.no_grad():
    for start_idx in tqdm(range(0, len(protein_ids), batch_size)):
        end_idx = min(start_idx + batch_size, len(protein_ids))

        # Get batch
        batch_emb = test_embeddings[start_idx:end_idx]
        batch_ids = protein_ids[start_idx:end_idx]

        # Predict
        batch_tensor = torch.FloatTensor(batch_emb).cuda()
        outputs = model(batch_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()

        # Process each protein in batch
        rows = []
        for i, pid in enumerate(batch_ids):
            protein_probs = probs[i]

            # Get predictions above threshold
            pred_indices = np.where(protein_probs > threshold)[0]

            # If no predictions, use top 3
            if len(pred_indices) == 0:
                no_pred_count += 1
                pred_indices = np.argsort(protein_probs)[-3:]

            # Create TSV rows for this protein (tab separated)
            for idx in pred_indices:
                go_term = idx_to_go_term[idx]
                confidence = float(protein_probs[idx])
                rows.append(f"{pid}\t{go_term}\t{confidence:.6f}\n")
                total_preds += 1

        # Write batch to file immediately
        with open(output_file, 'a') as f:
            f.writelines(rows)

        # Clear memory aggressively
        del batch_tensor, outputs, probs, rows
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"✓ Submission created: {output_file}")
print(f"  Total predictions: {total_preds}")
print(f"  Proteins with no predictions: {no_pred_count}")
print(f"{'='*60}")
```

---

## Step 12: Validate Submission

```python
# Cell 9: Validate submission file

import pandas as pd

# Read submission (TSV format - tab separated)
submission = pd.read_csv('/kaggle/working/submission.tsv', sep='\t', nrows=20)
print(f"First 20 rows:")
print(submission)

# Check stats
submission_full = pd.read_csv('/kaggle/working/submission.tsv', sep='\t')
print(f"\n{'='*60}")
print(f"Submission stats:")
print(f"  Total rows: {len(submission_full)}")
print(f"  Unique proteins: {submission_full['Protein ID'].nunique()}")
print(f"  Unique GO terms: {submission_full['GO Term'].nunique()}")
print(f"  Avg predictions per protein: {len(submission_full) / submission_full['Protein ID'].nunique():.1f}")

# Verify all test proteins have predictions
test_protein_count = len(protein_ids)
submitted_proteins = submission_full['Protein ID'].nunique()

if submitted_proteins < test_protein_count:
    print(f"⚠ Warning: {test_protein_count - submitted_proteins} proteins missing!")
else:
    print(f"✓ All {test_protein_count} test proteins have predictions")

print(f"{'='*60}")

# Check confidence distribution
print(f"\nConfidence distribution:")
print(submission_full['Confidence'].describe())

# Load sample submission to check format
sample_sub = pd.read_csv(
    '/kaggle/input/cafa-6-protein-function-prediction/sample_submission.tsv',
    sep='\t',
    nrows=10
)

print(f"\nSample submission format:")
print(sample_sub.head())

print(f"\nYour submission format:")
print(submission.head())

# Verify columns match
required_cols = ['Protein ID', 'GO Term', 'Confidence']
missing_cols = [col for col in required_cols if col not in submission.columns]
if missing_cols:
    print(f"\n❌ Missing columns: {missing_cols}")
else:
    print(f"\n✓ All required columns present")
    print(f"✓ Submission file ready for upload!")
print(f"\n✓ Submission file size: {os.path.getsize('/kaggle/working/submission.tsv') / (1024**2):.2f} MB")
```

**Note**: For a complete working Kaggle notebook, see [`baseline_submission.ipynb`](baseline_submission.ipynb) in this repo.

---

## Step 13: Submit to Kaggle

### Option A: Submit via Kaggle Notebook Output

1. **In your notebook**, make sure `submission.tsv` is in `/kaggle/working/`

2. **Click "Save Version"** (top right)
   - Select "Save & Run All"
   - Wait for notebook to finish (~2.5 hours total)

3. **After notebook completes**:
   - Click on "Output" tab
   - Find `submission.tsv`
   - Click "Submit to Competition"
   - Or download and upload manually

### Option B: Submit via Kaggle Website

1. **Download submission file** from notebook:
   ```python
   from IPython.display import FileLink
   display(FileLink('/kaggle/working/submission.tsv'))
   ```

2. **Go to competition page**:
   - https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/submit

3. **Upload `submission.tsv`**

4. **Add description**: "Baseline with ESM-2 150M + Simple Dense NN"

5. **Submit!**

### Option C: Submit via Kaggle API

```python
# Cell 12: Submit via API (if you have kaggle.json configured)

!pip install kaggle -q

# Submit
!kaggle competitions submit \
    -c cafa-6-protein-function-prediction \
    -f /kaggle/working/submission.tsv \
    -m "Baseline: ESM-2 150M + Dense NN (10 epochs, Focal Loss)"

print("\n✅ Submission uploaded!")
print("Check leaderboard: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/leaderboard")
```

---

## Complete Kaggle Workflow Summary

```python
# COMPLETE PIPELINE IN ONE NOTEBOOK

# 1. Setup (5 min)
!pip install fair-esm -q
# ... import libraries ...

# 2. Generate Train Embeddings (2 hours) ⏰
# ... ESM-2 embedding generation ...

# 3. Preprocess Labels (5 min)
# ... create label matrices ...

# 4. Train Model (30 min)
# ... train baseline classifier ...

# 5. Generate Test Embeddings (30 min)
# ... ESM-2 on test set ...

# 6. Generate Predictions (5 min)
# ... run inference ...

# 7. Create Submission (2 min)
# ... format predictions ...

# 8. Validate & Submit (1 min)
# ... check format and submit ...

# TOTAL TIME: ~3.5 hours
```

---

## Tips for Better Submissions

### 1. Tune the Threshold

```python
# Try different thresholds
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # Generate submission with this threshold
    # Submit and compare scores
    pass
```

Lower threshold → More predictions → Higher recall, lower precision
Higher threshold → Fewer predictions → Higher precision, lower recall

**Recommendation**: Start with 0.3, then tune based on leaderboard feedback.

### 2. Ensemble Multiple Models

```python
# Average predictions from multiple models
preds_model1 = ...  # Baseline
preds_model2 = ...  # Multi-task
preds_model3 = ...  # ESM-2 650M

# Weighted average
final_preds = 0.4 * preds_model1 + 0.3 * preds_model2 + 0.3 * preds_model3
```

### 3. Use Test-Time Augmentation

```python
# Generate embeddings with different truncation strategies
embeddings_balanced = generate_embeddings(strategy='balanced')
embeddings_start = generate_embeddings(strategy='start')

# Average predictions
preds_final = (preds_balanced + preds_start) / 2
```

### 4. Post-Process with GO Hierarchy

```python
# If you predict a child term, also predict parent terms
# This can improve consistency

# Load GO hierarchy from go-basic.obo
# Add parent terms to predictions
```

---

## Expected Submission Scores

With this baseline (ESM-2 150M, 10 epochs):

| Metric | Expected Score |
|--------|---------------|
| **F1 Score** | 0.45-0.55 |
| **Precision** | 0.40-0.50 |
| **Recall** | 0.50-0.60 |

This should place you in:
- **Top 50-70%** of submissions (decent baseline)
- Room for improvement with multi-task, ensembling, etc.

To improve:
1. Train multi-task model → +0.05 F1
2. Use ESM-2 650M → +0.03 F1
3. Optimize thresholds → +0.02 F1
4. Ensemble 3+ models → +0.05 F1
5. Fine-tune ESM-2 → +0.08 F1

**Potential final score**: 0.65-0.75 F1 (competitive!)

---

## Troubleshooting Submission Issues

### "File format not recognized"

Check file format:
```python
# Make sure it's TSV (tab-separated)
df.to_csv('submission.tsv', sep='\t', index=False)

# NOT CSV (comma-separated)
# df.to_csv('submission.csv', index=False)  # Wrong!
```

### "Missing predictions for some proteins"

```python
# Ensure all test proteins have predictions
test_protein_set = set(test_emb_data['protein_ids'])
submission_protein_set = set(submission_df['EntryID'])

missing = test_protein_set - submission_protein_set
print(f"Missing: {len(missing)} proteins")

# Add dummy predictions for missing proteins
for protein_id in missing:
    submission_df = submission_df.append({
        'EntryID': protein_id,
        'term': 'GO:0005515',  # Most common term
        'confidence': 0.1
    }, ignore_index=True)
```

### "Confidence scores out of range"

```python
# Clip confidence scores to [0, 1]
submission_df['confidence'] = submission_df['confidence'].clip(0, 1)
```

### "Too many predictions"

```python
# Limit to top K per protein
MAX_PREDICTIONS_PER_PROTEIN = 100

submission_df = (submission_df
    .sort_values(['EntryID', 'confidence'], ascending=[True, False])
    .groupby('EntryID')
    .head(MAX_PREDICTIONS_PER_PROTEIN)
    .reset_index(drop=True)
)
```

---

## Save Embeddings as Kaggle Dataset (Bonus Tip!)

After generating embeddings, save them as a dataset to reuse:

```python
# In your notebook, after generating embeddings:

# 1. Embeddings are in /kaggle/working/train_embeddings.pkl
# 2. Click "Save Version"
# 3. After notebook finishes, go to "Output" tab
# 4. Click "..." next to train_embeddings.pkl
# 5. Click "Add to Dataset"
# 6. Create new dataset: "cafa6-esm2-150m-embeddings"

# In future notebooks:
# Add Data → Your Datasets → cafa6-esm2-150m-embeddings
# Now you can load embeddings in 2 minutes instead of generating for 2 hours!
```

---

## Final Checklist

Before submitting:

- [ ] Generated train embeddings ✓
- [ ] Preprocessed labels ✓
- [ ] Trained model (F1 > 0.45) ✓
- [ ] Generated test embeddings ✓
- [ ] Created predictions ✓
- [ ] Formatted submission (TSV, 3 columns) ✓
- [ ] Validated submission (no errors) ✓
- [ ] Checked file size (< 500 MB) ✓
- [ ] All test proteins have predictions ✓
- [ ] Confidence scores in [0, 1] ✓
- [ ] Downloaded/submitted file ✓

---

## Next Steps After First Submission

1. **Check leaderboard** - How does your baseline compare?

2. **Improve model**:
   - Train multi-task model
   - Use ESM-2 650M embeddings
   - Ensemble multiple models
   - Optimize thresholds

3. **Iterate quickly**:
   - Save embeddings as dataset
   - Focus on model architecture and training
   - Don't regenerate embeddings each time!

4. **Track experiments**:
   - Keep notes of what works
   - Version your submissions
   - Compare scores

---

## 🎉 You're Ready!

You now have everything to:
1. ✅ Generate embeddings on Kaggle
2. ✅ Train a baseline model
3. ✅ Make predictions on test set
4. ✅ Create submission file
5. ✅ Submit to competition

**Good luck! 🚀**

---

*For questions or issues, refer back to the main documentation in the repository.*
