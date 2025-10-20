# Technical Notes - CAFA-6 Baseline

Advanced technical details for model optimization, sequence handling, and GPU configuration.

---

## Table of Contents

1. [Dataset Analysis](#dataset-analysis)
2. [ESM-2 Model Comparison](#esm-2-model-comparison)
3. [Long Sequence Handling](#long-sequence-handling)
4. [GPU Optimization (Kaggle)](#gpu-optimization-kaggle)
5. [Model Architectures](#model-architectures)
6. [Loss Functions](#loss-functions)
7. [Training Strategies](#training-strategies)

---

## Dataset Analysis

### Overview
- **82,404 proteins** in training set
- **26,125 unique GO terms** (labels)
- **537,027 total annotations**
- **Average 6.52 labels per protein** (range: 1-233)
- **Sequence lengths**: 3 to 35,213 amino acids (median: 376 AA)

### Aspect Distribution

| Aspect | GO Terms | Proteins | Avg Labels/Protein |
|--------|----------|----------|--------------------|
| **C** (Cellular Component) | 2,651 | 75,892 (92.1%) | 1.95 |
| **F** (Molecular Function) | 6,616 | 81,629 (99.1%) | 3.11 |
| **P** (Biological Process) | 16,858 | 78,336 (95.1%) | 3.57 |

**Key Insight**: 44.6% of proteins have all three aspects (C+F+P), suggesting multi-task learning is beneficial.

### Class Imbalance (Critical Challenge)

```
Label Frequency Distribution:
- Ultra-rare (1-5 proteins):    29.4% of labels
- Rare (6-10 proteins):         22.2% of labels
- Uncommon (11-50 proteins):    28.1% of labels
- Common (51-500 proteins):     15.8% of labels
- Very common (>500 proteins):   4.5% of labels
```

**51.6% of labels appear in fewer than 10 proteins!**

**Top 5 Most Common GO Terms**:
1. `GO:0005515` (protein binding) - 40.9% of proteins
2. `GO:0005737` (cytoplasm) - 37.2%
3. `GO:0005634` (nucleus) - 35.8%
4. `GO:0016020` (membrane) - 31.4%
5. `GO:0046872` (metal ion binding) - 28.7%

### Sequence Length Distribution

```
Percentiles:
- 25th: 197 AA
- 50th: 376 AA
- 75th: 664 AA
- 90th: 1,114 AA
- 95th: 1,589 AA
- 99th: 3,128 AA
```

**24.8% of sequences exceed 1024 AA** (ESM-2's max context length)

---

## ESM-2 Model Comparison

### Available Models

| Model | Params | Layers | Embed Dim | Speed | GPU Memory | Quality |
|-------|--------|--------|-----------|-------|------------|---------|
| esm2_t6_8M_UR50D | 8M | 6 | 320 | ⚡⚡⚡⚡⚡ | 2 GB | 70% |
| esm2_t12_35M_UR50D | 35M | 12 | 480 | ⚡⚡⚡⚡ | 4 GB | 80% |
| **esm2_t30_150M_UR50D** | **150M** | **30** | **640** | **⚡⚡⚡** | **8 GB** | **90%** ⭐ |
| esm2_t33_650M_UR50D | 650M | 33 | 1280 | ⚡⚡ | 12 GB | 100% |
| esm2_t36_3B_UR50D | 3B | 36 | 2560 | ⚡ | 24 GB | 105% |

### Recommendation

**For Baseline**: Use **esm2_t30_150M_UR50D** (150M)
- ✅ 90% of 650M performance
- ✅ 2x faster embedding generation
- ✅ Half the file size (21 GB vs 42 GB)
- ✅ Fits comfortably in Kaggle GPU (16 GB)

**For Final Model**: Upgrade to **esm2_t33_650M_UR50D** (650M)
- Expected +5-8% F1 improvement
- Takes ~2x longer (3-4 hours vs 1.5-2 hours)
- Better for competition leaderboard

### Embedding Generation Time (82k proteins)

| Model | P100 | T4 x2 (DataParallel) | File Size |
|-------|------|----------------------|-----------|
| 150M | 1.5-2 hours | 1.0-1.2 hours | ~21 GB |
| 650M | 3-4 hours | 2.0-2.5 hours | ~42 GB |
| 3B | 8-10 hours | 5-6 hours | ~84 GB |

---

## Long Sequence Handling

**Problem**: 24.8% of sequences exceed 1024 amino acids (ESM-2's max context)

### Strategy 1: Balanced Truncation (Recommended for Baseline)

**Keep both N-terminus and C-terminus:**
```python
if len(seq) > 1024:
    half = 1024 // 2
    truncated = seq[:half] + seq[-half:]
```

**Pros**:
- ✅ Preserves signal peptides (N-term)
- ✅ Preserves localization signals (C-term)
- ✅ Fast (no extra computation)
- ✅ Simple to implement

**Cons**:
- ❌ Loses middle domains

**Expected Performance**: 90-95% of full sequence performance

**Implementation**:
```python
from src.utils import LongSequenceHandler

handler = LongSequenceHandler()
truncated_seq = handler.truncate_balanced(sequence, max_length=1024)
```

### Strategy 2: Sliding Window with Pooling (Best Performance)

**Process overlapping chunks and average embeddings:**
```python
# Example: 2000 AA sequence with window=1024, stride=512
chunks = [seq[0:1024], seq[512:1536], seq[976:2000]]
embeddings = [embed(chunk) for chunk in chunks]
final_embedding = mean_pool(embeddings)
```

**Pros**:
- ✅ Captures full sequence information
- ✅ Best performance (+2-3% F1 over truncation)

**Cons**:
- ❌ Slower (3-4x for 2000 AA sequence)
- ❌ More complex

**Expected Performance**: 98-100% of full sequence performance

**Implementation**:
```python
from src.utils import MultiChunkEmbedder

embedder = MultiChunkEmbedder(model, alphabet)
embedding = embedder.embed_sequence(
    sequence,
    window_size=1024,
    stride=512,
    pooling='mean'  # or 'max', 'cls'
)
```

### Strategy 3: Hierarchical Chunks

**Process sequence in non-overlapping chunks, then aggregate:**
```python
chunks = [seq[i:i+1024] for i in range(0, len(seq), 1024)]
embeddings = [embed(chunk) for chunk in chunks]
final = attention_pooling(embeddings)  # Learned weights
```

**Use case**: Very long sequences (>3000 AA)

### Comparison

| Strategy | Speed | Performance | Complexity | Use Case |
|----------|-------|-------------|------------|----------|
| Balanced Truncation | ⚡⚡⚡ | 90-95% | Simple | Baseline |
| Sliding Window | ⚡ | 98-100% | Moderate | Final model |
| Hierarchical | ⚡⚡ | 95-98% | Complex | Very long seqs |

**Recommendation**: Start with balanced truncation, upgrade to sliding window for final submission.

---

## GPU Optimization (Kaggle)

### P100 vs T4 x2

Kaggle offers two GPU options:

#### P100 (Single GPU) ⭐ Recommended for Simplicity

**Specs**:
- 16 GB GDDR5
- 732 GB/s memory bandwidth
- 9.3 TFLOPS (FP32)

**Pros**:
- ✅ Simpler code (no DataParallel)
- ✅ Higher memory bandwidth (better for transformers)
- ✅ Easier debugging

**Expected Time**: 1.5-2 hours (82k proteins, ESM-2 150M)

**Script**: `generate_embeddings_kaggle.py`

#### T4 x2 (Dual GPU) - For Maximum Speed

**Specs**:
- 2x 16 GB GDDR6 = 32 GB total
- 320 GB/s per GPU (640 GB/s total)
- 8.1 TFLOPS (FP32) per GPU

**Pros**:
- ✅ Faster with DataParallel (1.5-1.8x speedup)
- ✅ More total memory (32 GB)

**Cons**:
- ❌ Requires DataParallel implementation
- ❌ Lower bandwidth per GPU
- ❌ Parallelization overhead (~10-15%)

**Expected Time**: 1.0-1.2 hours (82k proteins, ESM-2 150M)

**Script**: `generate_embeddings_kaggle_dual_gpu.py`

### Why P100 is Often Better for ESM-2

ESM-2 embedding generation is **memory-bandwidth bound**, not compute-bound:

1. **Memory Bandwidth Matters**:
   - P100: 732 GB/s
   - T4: 320 GB/s
   - **P100 is 2.3x faster at memory access**

2. **Transformers Load Weights Frequently**:
   - ESM-2 constantly loads attention weights
   - Memory bandwidth is the bottleneck
   - Higher bandwidth = faster inference

3. **DataParallel Overhead**:
   - Splitting batches across GPUs has cost
   - GPU synchronization takes time
   - ~10-15% overhead

**Result**: P100's higher bandwidth often matches or beats T4 x2's parallel advantage!

### When to Use T4 x2

Use T4 x2 if:
- ✅ Using DataParallel (provided script)
- ✅ Using large batches (batch_size ≥ 8 per GPU)
- ✅ Want absolute fastest speed
- ✅ Planning to train models (T4 x2 better for training)

### DataParallel Implementation

```python
import torch.nn as nn

# Load model
model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D')

# Wrap with DataParallel
n_gpus = torch.cuda.device_count()
if n_gpus > 1:
    model = nn.DataParallel(model)
    effective_batch_size = batch_size * n_gpus
else:
    effective_batch_size = batch_size

model = model.cuda()
model.eval()

# Forward pass (automatically splits across GPUs)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
```

### Monitoring GPU Usage

```python
# Check GPUs are being used
!nvidia-smi

# P100 should show:
# GPU 0: ~12-14 GB used, ~95% utilization

# T4 x2 should show:
# GPU 0: ~10-12 GB used, ~85% utilization
# GPU 1: ~10-12 GB used, ~85% utilization

# If GPU 1 shows 0% on T4 x2, you're using the wrong script!
```

---

## Model Architectures

### 1. Simple Baseline Classifier

**Use case**: Fast baseline with pre-computed embeddings

```python
class SimpleProteinClassifier(nn.Module):
    def __init__(self, embedding_dim=640, hidden_dim=512, num_classes=26125):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)
```

**Training time**: ~30 minutes
**Expected F1**: 0.50-0.60 (with 150M embeddings)

### 2. Multi-Task Classifier (Recommended)

**Use case**: Better performance with aspect-specific learning

```python
class MultiTaskProteinClassifier(nn.Module):
    def __init__(self, embedding_dim=640, shared_dim=1024, aspect_dim=512):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Aspect-specific heads
        self.c_head = nn.Sequential(
            nn.Linear(shared_dim, aspect_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(aspect_dim, 2651)  # C terms
        )

        self.f_head = nn.Sequential(
            nn.Linear(shared_dim, aspect_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(aspect_dim, 6616)  # F terms
        )

        self.p_head = nn.Sequential(
            nn.Linear(shared_dim, aspect_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(aspect_dim, 16858)  # P terms
        )

    def forward(self, embeddings):
        shared = self.shared(embeddings)
        return {
            'C': self.c_head(shared),
            'F': self.f_head(shared),
            'P': self.p_head(shared)
        }
```

**Training time**: ~45 minutes
**Expected F1**: 0.55-0.65 (with 150M embeddings)

### 3. End-to-End ESM-2 Classifier

**Use case**: Fine-tune ESM-2 (slower but best performance)

```python
class ESM2Classifier(nn.Module):
    def __init__(self, model_name='esm2_t30_150M_UR50D', num_classes=26125):
        super().__init__()
        self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)

        # Freeze ESM-2 layers (optional)
        for param in self.esm2.parameters():
            param.requires_grad = False

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.esm2.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, tokens):
        # Get ESM-2 embeddings
        with torch.no_grad():  # If frozen
            results = self.esm2(tokens, repr_layers=[33])
        embeddings = results['representations'][33][:, 0, :]  # [CLS] token

        # Classify
        return self.classifier(embeddings)
```

**Training time**: ~2-3 hours
**Expected F1**: 0.60-0.70 (if fine-tuned)

---

## Loss Functions

### 1. Focal Loss (Recommended) ⭐

**Best for extreme class imbalance**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()
```

**Why it works**:
- Focuses on hard-to-learn examples
- Down-weights easy negatives (important with 26k labels!)
- `gamma=2` focuses more on mistakes
- `alpha=0.25` balances pos/neg

**Expected improvement**: +5-10% F1 over BCE

### 2. Weighted BCE

**Simpler alternative to focal loss**

```python
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights  # Shape: (num_classes,)

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )
```

**Compute pos_weights**:
```python
# Higher weight for rare labels
pos_counts = labels.sum(axis=0)
neg_counts = len(labels) - pos_counts
pos_weights = neg_counts / (pos_counts + 1)  # +1 to avoid division by zero
```

### 3. Multi-Task Loss with Learnable Weights

**For multi-task learning**

```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable task weights (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(3))  # C, F, P

    def forward(self, predictions, targets, aspect_losses):
        # Compute weighted loss
        total_loss = 0
        for i, (pred, target, loss_fn) in enumerate(zip(predictions, targets, aspect_losses)):
            aspect_loss = loss_fn(pred, target)
            # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * aspect_loss + self.log_vars[i]
        return total_loss
```

**Benefits**:
- Automatically balances C/F/P tasks
- Learns which aspect needs more attention
- Better than fixed weights

---

## Training Strategies

### 1. Learning Rate Scheduling

**ReduceLROnPlateau (Recommended)**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize F1
    factor=0.5,
    patience=2,
    verbose=True
)
```

**Cosine Annealing**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

### 2. Threshold Optimization

**Problem**: Default threshold (0.5) is suboptimal

**Solution**: Find best threshold per GO term on validation set

```python
from sklearn.metrics import f1_score

def optimize_thresholds(predictions, targets, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresholds = []
    for i in range(targets.shape[1]):  # For each GO term
        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            binary_preds = (predictions[:, i] > t).astype(int)
            f1 = f1_score(targets[:, i], binary_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    return np.array(best_thresholds)
```

**Expected improvement**: +3-5% F1

### 3. Class Weights

**Compute inverse frequency weights**:
```python
def compute_class_weights(labels):
    """
    Args:
        labels: (num_proteins, num_classes) binary matrix
    Returns:
        weights: (num_classes,) array of weights
    """
    pos_counts = labels.sum(axis=0)
    weights = len(labels) / (pos_counts + 1)
    # Normalize to [1, max_weight]
    weights = weights / weights.min()
    # Cap at 10x to avoid over-weighting ultra-rare labels
    weights = np.minimum(weights, 10.0)
    return weights
```

### 4. Gradient Accumulation (for limited GPU memory)

```python
accumulation_steps = 4  # Effective batch size = batch_size * 4

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Early Stopping

```python
best_f1 = 0
patience = 5
no_improve = 0

for epoch in range(epochs):
    val_f1 = validate(model, val_loader)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 6. Ensemble

**Average predictions from multiple models**:

```python
# Train multiple models
models = [
    train(esm2_150M, focal_loss, seed=42),
    train(esm2_150M, focal_loss, seed=123),
    train(esm2_650M, focal_loss, seed=42)
]

# Ensemble predictions
predictions = []
for model in models:
    pred = model.predict(test_data)
    predictions.append(pred)

ensemble_pred = np.mean(predictions, axis=0)  # Average
```

**Expected improvement**: +3-5% F1

---

## Performance Optimization Summary

| Technique | Effort | F1 Gain | Time Cost |
|-----------|--------|---------|-----------|
| **ESM-2 150M → 650M** | Low | +5-8% | +2x time |
| **Focal Loss** | Low | +5-10% | None |
| **Multi-Task Learning** | Medium | +5-10% | +50% time |
| **Threshold Optimization** | Low | +3-5% | None |
| **Sliding Window** | Medium | +2-3% | +3x time |
| **Ensemble (3 models)** | High | +3-5% | +3x time |

**Recommended progression**:
1. Start: Baseline (ESM-2 150M + simple classifier) → **F1: 0.45-0.50**
2. Add focal loss + multi-task → **F1: 0.50-0.55**
3. Upgrade to ESM-2 650M → **F1: 0.55-0.60**
4. Add threshold optimization + sliding window → **F1: 0.58-0.63**
5. Ensemble 3 models → **F1: 0.60-0.65** (competitive for top 20%)

---

## Quick Reference

### Best Practices
- ✅ Start with ESM-2 150M (faster iteration)
- ✅ Use focal loss (handles imbalance)
- ✅ Multi-task learning (separate C/F/P heads)
- ✅ Balanced truncation for baseline
- ✅ P100 GPU on Kaggle (simpler)
- ✅ Save embeddings as dataset (reuse for training)
- ✅ Optimize thresholds on validation set

### Common Pitfalls
- ❌ Using default BCE loss (poor with imbalance)
- ❌ Not handling long sequences (loses info)
- ❌ Training without validation set (overfitting)
- ❌ Using fixed 0.5 threshold (suboptimal)
- ❌ Not monitoring GPU usage (wasting resources)
- ❌ Forgetting to save embeddings (regenerating wastes time)

---

For step-by-step Kaggle workflow, see [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).
