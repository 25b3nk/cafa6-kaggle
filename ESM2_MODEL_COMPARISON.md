# ESM-2 Model Comparison Guide

## Available ESM-2 Models

| Model Name | Parameters | Layers | Embedding Dim | Speed | File Size | GPU Memory | Performance |
|------------|-----------|---------|---------------|-------|-----------|------------|-------------|
| `esm2_t6_8M_UR50D` | 8M | 6 | **320** | ⚡⚡⚡⚡⚡ | ~400 MB | ~2 GB | 70% |
| `esm2_t12_35M_UR50D` | 35M | 12 | **480** | ⚡⚡⚡⚡ | ~1.2 GB | ~4 GB | 80% |
| `esm2_t30_150M_UR50D` | 150M | 30 | **640** | ⚡⚡⚡ | ~2.5 GB | ~8 GB | 90% |
| `esm2_t33_650M_UR50D` | 650M | 33 | **1280** | ⚡⚡ | ~10 GB | ~12 GB | **95%** |
| `esm2_t36_3B_UR50D` | 3B | 36 | **2560** | ⚡ | ~40 GB | ~24 GB | 100% |

*Performance percentages are relative to the largest model*

---

## Detailed Comparison

### ESM-2 8M (Smallest, Fastest) ⚡⚡⚡⚡⚡

```python
model_name = 'esm2_t6_8M_UR50D'
embedding_dim = 320
```

**Pros:**
- ✅ **Very fast**: 5-10x faster than 650M model
- ✅ **Low memory**: Fits easily in any GPU (2GB)
- ✅ **Small embeddings**: 320-dim → ~10 GB total file size
- ✅ **Quick iterations**: Perfect for experimentation
- ✅ **Works on CPU**: Reasonable speed even without GPU

**Cons:**
- ❌ Lower performance (~70% of best model)
- ❌ Less protein knowledge
- ❌ May struggle with complex/rare proteins

**Time Estimates (82k proteins):**
- Embedding generation: **~30-40 minutes** on GPU
- Embedding file size: **~10 GB**
- Training: **~15 minutes** per epoch

**Best for:**
- 🔬 Quick prototyping
- 🚀 Fast iteration on model architecture
- 💻 CPU-only environments
- 📊 Testing data pipelines

---

### ESM-2 35M (Small) ⚡⚡⚡⚡

```python
model_name = 'esm2_t12_35M_UR50D'
embedding_dim = 480
```

**Pros:**
- ✅ **Fast**: 3-5x faster than 650M model
- ✅ **Good performance**: ~80% of best model
- ✅ **Reasonable memory**: 4GB GPU sufficient
- ✅ **Medium file size**: 480-dim → ~15 GB total

**Cons:**
- ❌ Still lower than larger models
- ❌ May need more training to compensate

**Time Estimates (82k proteins):**
- Embedding generation: **~1-1.5 hours** on GPU
- Embedding file size: **~15 GB**
- Training: **~20 minutes** per epoch

**Best for:**
- 🎯 Good baseline with reasonable speed
- 📈 Balance between speed and performance
- 🔄 Multiple experiments in one day

---

### ESM-2 150M (Medium) ⚡⚡⚡

```python
model_name = 'esm2_t30_150M_UR50D'
embedding_dim = 640
```

**Pros:**
- ✅ **Strong performance**: ~90% of best model
- ✅ **Moderate speed**: 2x faster than 650M
- ✅ **Good balance**: Performance vs computational cost
- ✅ **Reasonable file size**: 640-dim → ~21 GB total

**Cons:**
- ❌ Still not as good as 650M model
- ❌ Requires 8GB GPU memory

**Time Estimates (82k proteins):**
- Embedding generation: **~1.5-2 hours** on GPU
- Embedding file size: **~21 GB**
- Training: **~25 minutes** per epoch

**Best for:**
- ⚖️ **Sweet spot for most users**
- 🏆 Competitive performance with reasonable resources
- 💾 Limited storage (smaller than 650M embeddings)

**Recommended if:** You want good performance but have storage/memory constraints

---

### ESM-2 650M (Large) ⚡⚡ ⭐ RECOMMENDED

```python
model_name = 'esm2_t33_650M_UR50D'
embedding_dim = 1280
```

**Pros:**
- ✅ **Excellent performance**: ~95% of best model
- ✅ **Industry standard**: Most papers use this
- ✅ **Best value**: Performance/size trade-off
- ✅ **Fits in Kaggle**: 16GB GPU is sufficient

**Cons:**
- ❌ Slower: ~3 hours for embedding generation
- ❌ Large files: ~42 GB embeddings
- ❌ Needs 12GB+ GPU memory

**Time Estimates (82k proteins):**
- Embedding generation: **~3 hours** on GPU
- Embedding file size: **~42 GB**
- Training: **~30 minutes** per epoch

**Best for:**
- 🏅 **Baseline for comparison**
- 📊 Final submission models
- 🎯 When you want best pre-trained knowledge
- 📄 Replicating published results

**Recommended if:** This is your main/final model

---

### ESM-2 3B (Huge) ⚡

```python
model_name = 'esm2_t36_3B_UR50D'
embedding_dim = 2560
```

**Pros:**
- ✅ **Best performance**: State-of-the-art
- ✅ **Most protein knowledge**

**Cons:**
- ❌ **Very slow**: ~12 hours for embeddings
- ❌ **Huge files**: ~150 GB embeddings
- ❌ **Requires 24GB+ GPU** (A100, won't fit on Kaggle)
- ❌ **Overkill** for most tasks

**Time Estimates (82k proteins):**
- Embedding generation: **~10-12 hours** on A100
- Embedding file size: **~150 GB**
- Training: **~1 hour** per epoch

**Best for:**
- 🔬 Research
- 💰 When you have A100/V100 GPUs
- 🏆 Squeezing last 1-2% performance

**Recommended if:** You have unlimited resources and need SOTA

---

## My Recommendations

### For Quick Start / Prototyping

**Use: ESM-2 8M** (`esm2_t6_8M_UR50D`)

```bash
python scripts/generate_embeddings.py \
    --model_name esm2_t6_8M_UR50D \
    --batch_size 16 \
    --strategy balanced
```

**Why:**
- Get embeddings in 30 minutes
- Test entire pipeline quickly
- Iterate on model architecture
- Verify everything works

**Timeline:** Day 1, first few hours

---

### For Good Baseline

**Use: ESM-2 150M** (`esm2_t30_150M_UR50D`)

```bash
python scripts/generate_embeddings.py \
    --model_name esm2_t30_150M_UR50D \
    --batch_size 8 \
    --strategy balanced
```

**Why:**
- 90% performance of 650M model
- 2x faster embedding generation
- Half the storage (21 GB vs 42 GB)
- Good enough for competition

**Timeline:** Day 1-2, establish solid baseline

---

### For Final Submission

**Use: ESM-2 650M** (`esm2_t33_650M_UR50D`)

```bash
python scripts/generate_embeddings.py \
    --model_name esm2_t33_650M_UR50D \
    --batch_size 8 \
    --strategy balanced
```

**Why:**
- Best performance/resource trade-off
- Industry standard (papers use this)
- Fits on Kaggle GPU
- 95% of SOTA performance

**Timeline:** Week 2-3, optimize submission

---

## Practical Strategy: Use Multiple Models

### Week 1: Quick Start
```bash
# Day 1: Quick prototype (30 min)
esm2_t6_8M_UR50D → Test pipeline, experiment with architectures

# Day 2-3: Solid baseline (2 hours)
esm2_t30_150M_UR50D → Establish good baseline, tune hyperparameters
```

### Week 2: Optimize
```bash
# Day 4-7: Best single model (3 hours)
esm2_t33_650M_UR50D → Train best model, threshold optimization
```

### Week 3: Ensemble
```bash
# Day 8+: Ensemble for final boost
Combine predictions from:
  - esm2_t30_150M_UR50D (diversity)
  - esm2_t33_650M_UR50D (performance)
  - Custom CNN model (different approach)
```

---

## File Size Comparison

### Embedding Files (82,404 proteins)

| Model | Embed Dim | File Size | Relative |
|-------|-----------|-----------|----------|
| 8M | 320 | ~10 GB | 1x |
| 35M | 480 | ~15 GB | 1.5x |
| 150M | 640 | ~21 GB | 2x |
| **650M** | **1280** | **~42 GB** | **4x** |
| 3B | 2560 | ~150 GB | 15x |

*Calculation: 82,404 proteins × embedding_dim × 4 bytes (float32)*

---

## Performance Expectations

Based on similar protein function prediction tasks:

| Model | Expected F1 (Micro) | Expected F1 (Macro) |
|-------|---------------------|---------------------|
| 8M | 0.45-0.50 | 0.12-0.18 |
| 35M | 0.50-0.55 | 0.15-0.20 |
| 150M | 0.55-0.60 | 0.18-0.23 |
| **650M** | **0.58-0.65** | **0.20-0.27** |
| 3B | 0.60-0.68 | 0.22-0.30 |

*These are rough estimates for baseline models without fine-tuning*

With fine-tuning and optimization, add +0.05-0.10 to F1 scores.

---

## How to Choose

### Choose 8M if:
- ⏰ You want results TODAY
- 💻 You only have CPU or small GPU
- 🔬 You're prototyping/experimenting
- 🎯 You just want to test the pipeline

### Choose 150M if:
- ⚖️ You want good performance with limited resources
- 💾 You have storage constraints
- 🏃 You want to iterate quickly
- 🎯 This is your main baseline

### Choose 650M if:
- 🏆 You want competitive results
- 💪 You have Kaggle GPU access
- 📊 This is your final submission
- 🎯 You want industry-standard performance

### Choose 3B if:
- 💰 You have A100 GPU access
- 🔬 You're doing research
- 🏅 You need every last % of performance
- 🎯 Resources are not a constraint

---

## Switching Between Models

All our code supports any ESM-2 model. Just change the parameter:

```python
# In scripts/generate_embeddings.py
--model_name esm2_t6_8M_UR50D     # Fast
--model_name esm2_t30_150M_UR50D  # Balanced
--model_name esm2_t33_650M_UR50D  # Best

# In training code
from src.models import SimpleProteinClassifier

# For 8M embeddings
model = SimpleProteinClassifier(embedding_dim=320, ...)

# For 150M embeddings
model = SimpleProteinClassifier(embedding_dim=640, ...)

# For 650M embeddings
model = SimpleProteinClassifier(embedding_dim=1280, ...)
```

---

## My Final Recommendation

### Start Small, Scale Up

**Day 1: Use ESM-2 8M**
- Generate embeddings (30 min)
- Build and test pipeline (2 hours)
- Train baseline model (15 min)
- **Get F1 ~0.47** ✅

**Day 2-3: Use ESM-2 150M**
- Generate embeddings (2 hours)
- Train better model (30 min)
- Optimize thresholds (1 hour)
- **Get F1 ~0.57** ✅

**Week 2: Use ESM-2 650M**
- Generate embeddings (3 hours)
- Multi-task training (6 hours)
- Fine-tune (8 hours)
- **Get F1 ~0.63** ✅

**Week 3: Ensemble**
- Combine 150M + 650M predictions
- **Get F1 ~0.65+** 🏆

---

## Command Examples

### Generate All Model Embeddings

```bash
# Quick (8M)
python scripts/generate_embeddings.py \
    --model_name esm2_t6_8M_UR50D \
    --output_dir data/embeddings/8M \
    --batch_size 16

# Balanced (150M)
python scripts/generate_embeddings.py \
    --model_name esm2_t30_150M_UR50D \
    --output_dir data/embeddings/150M \
    --batch_size 8

# Best (650M)
python scripts/generate_embeddings.py \
    --model_name esm2_t33_650M_UR50D \
    --output_dir data/embeddings/650M \
    --batch_size 8
```

Then compare performance and decide!

---

## Bottom Line

**For this competition:**

1. **Prototype with 8M** (today)
2. **Baseline with 150M** (this week) ⭐ **START HERE**
3. **Optimize with 650M** (next week)
4. **Ensemble both** (final submission)

The **150M model gives you 90% performance for 50% effort** - that's the sweet spot! 🎯
