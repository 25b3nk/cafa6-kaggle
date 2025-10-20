# Scripts for CAFA-6 Baseline

This directory contains scripts for generating ESM-2 embeddings and training baseline models.

## ESM-2 Embedding Generation

### Local Environment

Generate embeddings on your local machine (if you have GPU):

```bash
# Install ESM-2
pip install fair-esm

# Generate embeddings for training set
python scripts/generate_embeddings.py \
    --data_dir cafa-6-protein-function-prediction \
    --output_dir data/embeddings \
    --split train \
    --model_name esm2_t33_650M_UR50D \
    --strategy balanced \
    --batch_size 8 \
    --device cuda

# Expected time: ~3 hours on GPU for 82k proteins
# Expected size: ~40-50 GB
```

**Arguments:**
- `--data_dir`: Path to competition data directory
- `--output_dir`: Where to save embeddings
- `--split`: 'train' or 'test'
- `--model_name`: ESM-2 model variant
  - `esm2_t6_8M_UR50D` (smallest, fastest)
  - `esm2_t12_35M_UR50D` (small)
  - `esm2_t30_150M_UR50D` (medium)
  - `esm2_t33_650M_UR50D` (large, **recommended**)
  - `esm2_t36_3B_UR50D` (huge, slow)
- `--strategy`: Truncation strategy
  - `balanced` - Keep N-term + C-term (**recommended**)
  - `start` - Keep N-terminus only
  - `end` - Keep C-terminus only
  - `middle` - Keep middle region
- `--batch_size`: Batch size (8-16 for GPU, 1-4 for CPU)
- `--device`: 'cuda' or 'cpu'

### Kaggle Environment

Use the Kaggle-optimized script in a notebook:

```python
# In Kaggle notebook
!pip install fair-esm

# Copy the script
!cp ../input/your-dataset/generate_embeddings_kaggle.py .

# Run it
from generate_embeddings_kaggle import generate_embeddings_kaggle

embeddings_path = generate_embeddings_kaggle(
    data_dir='/kaggle/input/cafa-6-protein-function-prediction',
    output_dir='/kaggle/working/embeddings',
    model_name='esm2_t33_650M_UR50D',
    split='train',
    batch_size=4,  # Smaller for Kaggle GPU
    strategy='balanced'
)

# Expected time: ~3-4 hours on Kaggle GPU
```

**Kaggle-specific optimizations:**
- Smaller batch sizes (4-8) to fit in GPU memory
- Saves in chunks to avoid memory issues
- Automatic garbage collection
- Works on P100 (16GB) and T4 (16GB) GPUs

## Testing Embeddings

Verify that embeddings were generated correctly:

```bash
python scripts/test_embeddings.py \
    data/embeddings/train_embeddings_esm2_t33_650M_UR50D_balanced.pkl
```

**Output:**
```
Testing embeddings from: data/embeddings/train_embeddings_esm2_t33_650M_UR50D_balanced.pkl
============================================================

✓ All required fields present

✓ Shapes are correct:
  Embeddings: (82404, 1280)
  Protein IDs: 82404

✓ Data types are correct:
  Embeddings dtype: float32

✓ No NaN or Inf values

✓ Embedding statistics:
  Mean: 0.0234
  Std: 1.4562
  Min: -8.4521
  Max: 9.1234

✓ All protein IDs are unique

✓ Sample embeddings:
  Q5W0B1: shape=(1280,), mean=0.0345, std=1.4123
  Q3EC77: shape=(1280,), mean=-0.0123, std=1.5234
  Q8IZR5: shape=(1280,), mean=0.0567, std=1.3987

✓ Metadata:
  Model: esm2_t33_650M_UR50D
  Embedding dim: 1280
  Max length: 1024
  Strategy: balanced
  Truncated: 20456 (24.8%)

============================================================
✓ ALL TESTS PASSED!
============================================================
```

### Compare Different Strategies

Compare embeddings generated with different truncation strategies:

```bash
python scripts/test_embeddings.py \
    data/embeddings/train_embeddings_balanced.pkl \
    --compare data/embeddings/train_embeddings_start.pkl
```

**Output:**
```
Cosine similarity (first 100 common proteins):
  Mean: 0.8234
  Std: 0.0567
  Min: 0.6123
  Max: 0.9876

✓ Embeddings are similar (different strategies but compatible)
```

## Model Selection

### ESM-2 Model Variants

| Model | Params | Layers | Embed Dim | Speed | Performance | GPU Memory |
|-------|--------|--------|-----------|-------|-------------|------------|
| esm2_t6_8M_UR50D | 8M | 6 | 320 | Very Fast | Good | ~2 GB |
| esm2_t12_35M_UR50D | 35M | 12 | 480 | Fast | Better | ~4 GB |
| esm2_t30_150M_UR50D | 150M | 30 | 640 | Medium | Great | ~8 GB |
| **esm2_t33_650M_UR50D** | 650M | 33 | **1280** | Slow | **Excellent** | ~12 GB |
| esm2_t36_3B_UR50D | 3B | 36 | 2560 | Very Slow | Best | ~24 GB |

**Recommendation**: Use `esm2_t33_650M_UR50D` (650M parameters)
- Best balance of performance and speed
- Fits in Kaggle GPU (16GB)
- Used in most papers
- 1280-dim embeddings

## Truncation Strategies

Based on our EDA, **24.8% of sequences are longer than 1024 AA**.

### Balanced Truncation (Recommended)

```python
# Keeps both N-terminus and C-terminus
# Example: 2000 AA sequence → [First 512] + [Last 512]
strategy='balanced'
```

**Pros:**
- Preserves signal peptides (N-term)
- Preserves localization signals (C-term)
- Works well for most proteins

**Cons:**
- Loses middle domains

**Use for:** Baseline model (fast, good performance)

### Sliding Window (Best Performance)

```python
# Processes overlapping chunks, averages embeddings
# Example: 2000 AA → 3 chunks with overlap
strategy='sliding_window'  # Not yet in generate_embeddings.py
```

**Note:** Sliding window requires custom implementation.
See `src/utils/sequence_handling.py` for details.

**Use for:** Final model (slower but better accuracy)

## Expected Output

After running embedding generation:

```
data/embeddings/
├── train_embeddings_esm2_t33_650M_UR50D_balanced.pkl  (~40-50 GB)
└── test_embeddings_esm2_t33_650M_UR50D_balanced.pkl   (~5-10 GB)
```

**Pickle file contents:**
```python
{
    'embeddings': np.array,          # (num_proteins, 1280)
    'protein_ids': list,             # List of protein IDs
    'embeddings_dict': dict,         # protein_id -> embedding
    'model_name': str,               # 'esm2_t33_650M_UR50D'
    'embedding_dim': int,            # 1280
    'max_length': int,               # 1024
    'strategy': str,                 # 'balanced'
    'num_truncated': int,            # Number of truncated sequences
    'num_proteins': int              # Total proteins
}
```

## Training Models

### Train Baseline Model

```bash
python scripts/train_baseline.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --output_dir models/baseline \
    --epochs 10 \
    --batch_size 32 \
    --loss focal \
    --lr 1e-4
```

**Key arguments:**
- `--hidden_dims`: Hidden layer sizes (default: `1024 512`)
- `--dropout`: Dropout rate (default: `0.3`)
- `--loss`: Loss function - `focal`, `bce`, or `weighted_bce`
- `--focal_alpha`: Focal loss alpha (default: `0.25`)
- `--focal_gamma`: Focal loss gamma (default: `2.0`)
- `--scheduler`: LR scheduler - `plateau`, `cosine`, `step`, or `none`

**Expected time:** ~30 minutes for 10 epochs on GPU

**Expected F1:** 0.50-0.60 with 150M embeddings

### Train Multi-Task Model

```bash
python scripts/train_multitask.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --output_dir models/multitask \
    --epochs 15 \
    --loss_type focal \
    --learnable_weights
```

**Key arguments:**
- `--shared_hidden_dims`: Shared encoder dims (default: `2048 1024`)
- `--aspect_hidden_dim`: Aspect head hidden dim (default: `512`)
- `--loss_type`: Loss for each aspect - `focal`, `bce`, or `asymmetric`
- `--learnable_weights`: Use learnable task weights (recommended)

**Expected time:** ~45 minutes for 15 epochs on GPU

**Expected F1:** 0.55-0.65 with 150M embeddings

## Evaluation

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model models/baseline/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --output_dir results/baseline \
    --threshold 0.5
```

**With threshold optimization:**
```bash
python scripts/evaluate.py \
    --model models/baseline/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --output_dir results/baseline \
    --optimize_threshold
```

This will find the best threshold for each GO term separately, which can improve F1 by 3-5%.

**For multi-task models:**
```bash
python scripts/evaluate.py \
    --model models/multitask/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --output_dir results/multitask \
    --multitask
```

## Complete Pipeline

Here's the full workflow from scratch:

```bash
# 1. Generate embeddings (~2 hours with 150M model)
python scripts/generate_embeddings.py \
    --data_dir cafa-6-protein-function-prediction \
    --output_dir data/embeddings \
    --model_name esm2_t30_150M_UR50D \
    --strategy balanced

# 2. Preprocess labels (~5 minutes)
python -c "
from src.data import ProteinDataPreprocessor
preprocessor = ProteinDataPreprocessor('cafa-6-protein-function-prediction')
preprocessor.preprocess_all(output_dir='data/processed')
"

# 3. Train baseline model (~30 minutes)
python scripts/train_baseline.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --epochs 10

# 4. Evaluate (~5 minutes)
python scripts/evaluate.py \
    --model models/baseline/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --optimize_threshold

# 5. Train multi-task model for better results (~45 minutes)
python scripts/train_multitask.py \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --epochs 15 \
    --learnable_weights

# 6. Evaluate multi-task
python scripts/evaluate.py \
    --model models/multitask/best_model.pt \
    --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
    --labels data/processed/preprocessed_data.pkl \
    --multitask
```

**Total time:** ~3-4 hours from scratch to competitive baseline!

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 2

# Or use smaller model
--model_name esm2_t30_150M_UR50D

# Or use CPU (very slow)
--device cpu
```

### File Too Large

Embeddings can be 40-50 GB. Options:

1. **Use smaller model:** `esm2_t30_150M_UR50D` (640-dim)
2. **Compress:** Use float16 instead of float32 (halves size)
3. **Split:** Generate embeddings in batches

### ImportError: fair-esm

```bash
pip install fair-esm
```

Or from GitHub:
```bash
pip install git+https://github.com/facebookresearch/esm.git
```

## Tips for Kaggle

1. **Save embeddings as Kaggle dataset:**
   - Generate embeddings once
   - Upload as private dataset
   - Reuse across notebooks (saves GPU time)

2. **Use chunked generation:**
   - Kaggle notebooks timeout after 9 hours
   - Use `generate_embeddings_kaggle.py` which saves in chunks
   - Can resume if interrupted

3. **GPU vs CPU:**
   - GPU: ~3 hours for 82k proteins
   - CPU: ~48 hours for 82k proteins
   - **Always use GPU on Kaggle!**

4. **Memory management:**
   - Clear cache: `torch.cuda.empty_cache()`
   - Use `gc.collect()` between chunks
   - Monitor with: `nvidia-smi`

## References

- ESM-2 Paper: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1
- ESM GitHub: https://github.com/facebookresearch/esm
- Model Zoo: https://github.com/facebookresearch/esm#available-models
