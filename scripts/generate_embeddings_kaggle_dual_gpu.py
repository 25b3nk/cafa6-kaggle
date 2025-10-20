"""
Kaggle-optimized ESM-2 embedding generation for DUAL GPU (T4 x2)
Uses DataParallel to leverage both GPUs for faster embedding generation

This version is optimized for T4 x2 setup and will be ~1.8x faster than single GPU
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import gc


def generate_embeddings_dual_gpu(data_dir: str,
                                 output_dir: str = '/kaggle/working/embeddings',
                                 model_name: str = 'esm2_t30_150M_UR50D',
                                 split: str = 'train',
                                 batch_size: int = 8,  # Larger batch for dual GPU
                                 max_length: int = 1024,
                                 strategy: str = 'balanced',
                                 chunk_size: int = 5000):
    """
    Generate ESM-2 embeddings using DUAL GPUs (T4 x2)

    Args:
        data_dir: Path to data directory
        output_dir: Where to save embeddings
        model_name: ESM-2 model variant
        split: 'train' or 'test'
        batch_size: Batch size (can be larger with dual GPU - try 8-12)
        max_length: Max sequence length
        strategy: Truncation strategy
        chunk_size: Save embeddings every N proteins
    """

    import sys
    sys.path.append('..')

    from src.data import ProteinDataPreprocessor
    from src.utils import LongSequenceHandler

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")

    if n_gpus == 0:
        print("❌ No GPU found! This script requires GPU.")
        return

    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Load sequences
    print("\nLoading sequences...")
    preprocessor = ProteinDataPreprocessor(data_dir)
    sequences = preprocessor.load_sequences(split)
    protein_ids = list(sequences.keys())

    print(f"Total proteins: {len(protein_ids)}")

    # Load ESM-2 model
    print(f"\nLoading ESM-2 model: {model_name}")
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()

    # IMPORTANT: Use DataParallel for multi-GPU
    if n_gpus > 1:
        print(f"🚀 Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
        # Adjust batch size for multi-GPU
        effective_batch_size = batch_size * n_gpus
        print(f"   Effective batch size: {effective_batch_size} ({batch_size} per GPU)")
    else:
        print("Using single GPU")
        effective_batch_size = batch_size

    model = model.cuda()
    model.eval()

    embedding_dim = model.module.embed_dim if n_gpus > 1 else model.embed_dim
    num_layers = model.module.num_layers if n_gpus > 1 else model.num_layers
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of layers: {num_layers}")

    # Truncation handler
    handler = LongSequenceHandler()

    def truncate_seq(seq):
        if len(seq) <= max_length:
            return seq
        if strategy == 'balanced':
            return handler.truncate_balanced(seq, max_length)
        elif strategy == 'start':
            return handler.truncate_start(seq, max_length)
        else:
            return seq[:max_length]

    # Generate embeddings in chunks
    all_embeddings = []
    all_protein_ids = []
    chunk_num = 0

    print(f"\nGenerating embeddings (batch_size={effective_batch_size}, chunk_size={chunk_size})...")

    for i in tqdm(range(0, len(protein_ids), effective_batch_size)):
        batch_ids = protein_ids[i:i + effective_batch_size]
        batch_data = [(pid, truncate_seq(sequences[pid])) for pid in batch_ids]

        # Convert batch
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.cuda()

        # Generate embeddings (use last layer)
        # With DataParallel, the model will automatically split the batch across GPUs
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)
            embeddings = results['representations'][num_layers][:, 0, :].cpu().numpy()

        all_embeddings.extend(embeddings)
        all_protein_ids.extend(batch_ids)

        # Save chunk and clear memory
        if len(all_embeddings) >= chunk_size:
            chunk_path = output_path / f'embeddings_chunk_{chunk_num}.pkl'
            chunk_data = {
                'embeddings': np.array(all_embeddings),
                'protein_ids': all_protein_ids
            }
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f)

            print(f"\nSaved chunk {chunk_num}: {len(all_embeddings)} proteins")

            # Clear memory
            all_embeddings = []
            all_protein_ids = []
            chunk_num += 1
            gc.collect()
            torch.cuda.empty_cache()

    # Save remaining embeddings
    if all_embeddings:
        chunk_path = output_path / f'embeddings_chunk_{chunk_num}.pkl'
        chunk_data = {
            'embeddings': np.array(all_embeddings),
            'protein_ids': all_protein_ids
        }
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        print(f"\nSaved final chunk {chunk_num}: {len(all_embeddings)} proteins")

    # Combine all chunks into single file
    print("\nCombining chunks...")
    all_embeddings = []
    all_protein_ids = []

    for chunk_idx in range(chunk_num + 1):
        chunk_path = output_path / f'embeddings_chunk_{chunk_idx}.pkl'
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
        all_embeddings.append(chunk_data['embeddings'])
        all_protein_ids.extend(chunk_data['protein_ids'])

    embeddings_array = np.vstack(all_embeddings)

    # Save combined file
    final_path = output_path / f'{split}_embeddings_{model_name}.pkl'
    final_data = {
        'embeddings': embeddings_array,
        'protein_ids': all_protein_ids,
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'max_length': max_length,
        'strategy': strategy,
        'num_proteins': len(all_protein_ids),
        'n_gpus': n_gpus
    }

    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"\n{'='*60}")
    print(f"Embeddings saved to: {final_path}")
    print(f"Shape: {embeddings_array.shape}")
    print(f"Size: {final_path.stat().st_size / (1024**3):.2f} GB")
    print(f"GPUs used: {n_gpus}")
    print(f"{'='*60}")

    return final_path


# Example usage in Kaggle notebook:
"""
# Check GPU configuration first
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# If you have 2 T4s, use this dual-GPU version
if torch.cuda.device_count() == 2:
    print("Using DUAL GPU version (faster!)")
    from generate_embeddings_kaggle_dual_gpu import generate_embeddings_dual_gpu

    embeddings_path = generate_embeddings_dual_gpu(
        data_dir='/kaggle/input/cafa-6-protein-function-prediction',
        output_dir='/kaggle/working/embeddings',
        model_name='esm2_t30_150M_UR50D',
        split='train',
        batch_size=8,  # Per GPU, so effective batch = 16
        strategy='balanced'
    )

# If you have 1 P100, use single GPU version
else:
    print("Using SINGLE GPU version")
    from generate_embeddings_kaggle import generate_embeddings_kaggle

    embeddings_path = generate_embeddings_kaggle(
        data_dir='/kaggle/input/cafa-6-protein-function-prediction',
        output_dir='/kaggle/working/embeddings',
        model_name='esm2_t30_150M_UR50D',
        split='train',
        batch_size=4,
        strategy='balanced'
    )

print(f"Done! Embeddings at: {embeddings_path}")
"""


# Benchmark helper
def benchmark_gpu_options():
    """
    Helper to compare P100 vs T4 x2 performance
    """
    import time

    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        print("No GPU available")
        return

    print(f"Benchmarking with {n_gpus} GPU(s)...")

    # Create dummy model
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D')

    num_layers = model.num_layers

    if n_gpus > 1:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    # Create dummy data
    dummy_sequences = [("protein1", "A" * 512) for _ in range(32)]
    _, _, batch_tokens = batch_converter(dummy_sequences)
    batch_tokens = batch_tokens.cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch_tokens, repr_layers=[num_layers])

    torch.cuda.synchronize()

    # Benchmark
    n_iterations = 20
    start = time.time()

    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(batch_tokens, repr_layers=[num_layers])

    torch.cuda.synchronize()
    end = time.time()

    time_per_batch = (end - start) / n_iterations
    throughput = 32 / time_per_batch  # proteins per second

    print(f"\nResults:")
    print(f"  Time per batch (32 proteins): {time_per_batch:.3f}s")
    print(f"  Throughput: {throughput:.1f} proteins/sec")
    print(f"  Estimated time for 82k proteins: {82000 / throughput / 60:.1f} minutes")

    return throughput


if __name__ == '__main__':
    # Run benchmark
    print("GPU Benchmark")
    print("="*60)
    benchmark_gpu_options()
