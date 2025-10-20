# Fix: RAM Overload During Submission Creation

## Problem

Your current code loads all predictions into RAM:
```python
all_predictions = []
for i in tqdm(range(0, len(test_embeddings_tensor), batch_size)):
    # ...
    all_predictions.append(probs)  # Accumulates in RAM!
predictions = np.vstack(all_predictions)  # OOM here!
```

With 26,125 GO terms and thousands of test proteins, this creates a massive array that overloads RAM.

---

## Solution: Stream to File

Instead of accumulating predictions, write them directly to the TSV file in batches.

### Replace Your Prediction Code With This:

```python
# Kaggle Notebook Cell - Memory-Efficient Submission Creation (TSV format)

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
threshold = 0.5
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

# Validate (TSV format - tab separated)
submission = pd.read_csv(output_file, sep='\t', nrows=20)
print(f"\nFirst 20 rows:")
print(submission)

# Check stats
submission_full = pd.read_csv(output_file, sep='\t')
print(f"\nSubmission stats:")
print(f"  Total rows: {len(submission_full)}")
print(f"  Unique proteins: {submission_full['Protein ID'].nunique()}")
print(f"  Unique GO terms: {submission_full['GO Term'].nunique()}")

# Verify all proteins have predictions
submitted_proteins = submission_full['Protein ID'].nunique()
if submitted_proteins < len(protein_ids):
    print(f"⚠ Warning: {len(protein_ids) - submitted_proteins} proteins missing!")
else:
    print(f"✓ All {len(protein_ids)} proteins have predictions")
```

---

## Key Changes

### Before (RAM Overload):
```python
all_predictions = []
for batch in batches:
    probs = predict(batch)
    all_predictions.append(probs)  # ❌ Accumulates in RAM

predictions = np.vstack(all_predictions)  # ❌ OOM!

# Then process predictions
for protein in proteins:
    # ...
```

**Memory usage**: `num_proteins × num_classes × 4 bytes`
- Example: 10,000 proteins × 26,125 classes × 4 bytes = **~1 GB per batch**

### After (Memory Efficient):
```python
with open(output_file, 'w') as f:
    f.write("header\n")

for batch in batches:
    probs = predict(batch)  # ✅ Small batch only

    # Process and write immediately
    rows = create_rows(probs)  # ✅ Small list
    with open(output_file, 'a') as f:
        f.writelines(rows)  # ✅ Write to disk

    del probs, rows  # ✅ Clear memory
    gc.collect()
```

**Memory usage**: `batch_size × num_classes × 4 bytes`
- Example: 16 proteins × 26,125 classes × 4 bytes = **~1.6 MB per batch** ✅

---

## Additional Optimizations

### 1. Reduce Batch Size
If still getting OOM:
```python
batch_size = 8  # Even smaller
# or
batch_size = 4  # Very conservative
```

### 2. Process One Protein at a Time (Slowest but safest)
```python
batch_size = 1

for i, pid in enumerate(tqdm(protein_ids)):
    emb = test_embeddings[i:i+1]
    tensor = torch.FloatTensor(emb).cuda()
    output = model(tensor)
    probs = torch.sigmoid(output).cpu().numpy()[0]

    # Process and write
    pred_indices = np.where(probs > threshold)[0]
    if len(pred_indices) == 0:
        pred_indices = np.argsort(probs)[-3:]

    rows = [f"{pid},{idx_to_go_term[idx]},{probs[idx]:.6f}\n"
            for idx in pred_indices]

    with open(output_file, 'a') as f:
        f.writelines(rows)

    del tensor, output, probs, rows
    if i % 100 == 0:  # Clear every 100 proteins
        gc.collect()
        torch.cuda.empty_cache()
```

### 3. Use Float16 (Half Precision)
```python
model = model.half()  # Convert to float16
batch_tensor = torch.FloatTensor(batch_emb).half().cuda()
```

Saves 50% memory, slight accuracy loss.

---

## Why This Works

1. **Streaming Write**: Write to disk immediately instead of accumulating in RAM
2. **Small Batches**: Process 16 proteins at a time (vs all at once)
3. **Aggressive Cleanup**: Delete and garbage collect after each batch
4. **No Large Arrays**: Never create the full `(num_proteins, num_classes)` array

---

## Expected Behavior

```
Generating predictions for 10000 proteins...
100%|████████████████████| 625/625 [02:30<00:00, 4.17it/s]

============================================================
✓ Submission created: /kaggle/working/submission.tsv
  Total predictions: 85432
  Proteins with no predictions: 23
============================================================

First 20 rows:
      Protein ID      GO Term  Confidence
0       Q9Y6K9  GO:0005515    0.876543
1       Q9Y6K9  GO:0005737    0.654321
...

Submission stats:
  Total rows: 85432
  Unique proteins: 10000
  Unique GO terms: 15234
✓ All 10000 proteins have predictions
```

---

## Summary

**Old approach**:
- Load all predictions → OOM ❌

**New approach**:
- Process small batch → Write to file → Clear memory → Repeat ✅

**Memory savings**: From ~1-2 GB to ~1-2 MB per batch (1000x reduction!)
