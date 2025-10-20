"""
Test script to verify ESM-2 embeddings are generated correctly
"""

import torch
import numpy as np
import pickle
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_embeddings(embeddings_path: str):
    """
    Test that embeddings are valid and properly formatted

    Args:
        embeddings_path: Path to embeddings pickle file
    """
    print(f"Testing embeddings from: {embeddings_path}")
    print("=" * 60)

    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    # Check required fields
    required_fields = ['embeddings', 'protein_ids', 'model_name', 'embedding_dim', 'num_proteins']
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    embeddings = data['embeddings']
    protein_ids = data['protein_ids']
    embedding_dim = data['embedding_dim']
    num_proteins = data['num_proteins']

    print(f"\n✓ All required fields present")

    # Check shapes
    assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
    assert len(embeddings.shape) == 2, "Embeddings should be 2D array"
    assert embeddings.shape[0] == num_proteins, "Number of embeddings doesn't match num_proteins"
    assert embeddings.shape[1] == embedding_dim, "Embedding dimension mismatch"
    assert len(protein_ids) == num_proteins, "Number of protein IDs doesn't match"

    print(f"\n✓ Shapes are correct:")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Protein IDs: {len(protein_ids)}")

    # Check data types
    assert embeddings.dtype in [np.float32, np.float64], "Embeddings should be float type"
    assert all(isinstance(pid, str) for pid in protein_ids[:10]), "Protein IDs should be strings"

    print(f"\n✓ Data types are correct:")
    print(f"  Embeddings dtype: {embeddings.dtype}")

    # Check for NaN or Inf
    num_nan = np.isnan(embeddings).sum()
    num_inf = np.isinf(embeddings).sum()

    assert num_nan == 0, f"Found {num_nan} NaN values in embeddings"
    assert num_inf == 0, f"Found {num_inf} Inf values in embeddings"

    print(f"\n✓ No NaN or Inf values")

    # Check embedding statistics
    emb_mean = embeddings.mean()
    emb_std = embeddings.std()
    emb_min = embeddings.min()
    emb_max = embeddings.max()

    print(f"\n✓ Embedding statistics:")
    print(f"  Mean: {emb_mean:.4f}")
    print(f"  Std: {emb_std:.4f}")
    print(f"  Min: {emb_min:.4f}")
    print(f"  Max: {emb_max:.4f}")

    # Check for reasonable values (ESM-2 embeddings typically in range [-10, 10])
    assert -50 < emb_mean < 50, f"Mean value seems unusual: {emb_mean}"
    assert emb_std > 0, "Standard deviation should be positive"

    # Check for duplicate protein IDs
    unique_ids = set(protein_ids)
    assert len(unique_ids) == len(protein_ids), "Found duplicate protein IDs"

    print(f"\n✓ All protein IDs are unique")

    # Sample a few embeddings
    print(f"\n✓ Sample embeddings:")
    for i in range(min(3, num_proteins)):
        pid = protein_ids[i]
        emb = embeddings[i]
        print(f"  {pid}: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}")

    # Check metadata
    print(f"\n✓ Metadata:")
    print(f"  Model: {data['model_name']}")
    print(f"  Embedding dim: {data['embedding_dim']}")
    print(f"  Max length: {data.get('max_length', 'N/A')}")
    print(f"  Strategy: {data.get('strategy', 'N/A')}")
    if 'num_truncated' in data:
        print(f"  Truncated: {data['num_truncated']} ({data['num_truncated']/num_proteins*100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


def compare_embeddings(path1: str, path2: str):
    """
    Compare two embedding files (useful for testing different strategies)

    Args:
        path1: First embeddings file
        path2: Second embeddings file
    """
    print(f"Comparing embeddings:")
    print(f"  File 1: {path1}")
    print(f"  File 2: {path2}")
    print("=" * 60)

    # Load both
    with open(path1, 'rb') as f:
        data1 = pickle.load(f)
    with open(path2, 'rb') as f:
        data2 = pickle.load(f)

    emb1 = data1['embeddings']
    emb2 = data2['embeddings']

    # Find common proteins
    ids1 = set(data1['protein_ids'])
    ids2 = set(data2['protein_ids'])

    common_ids = ids1 & ids2
    print(f"\nCommon proteins: {len(common_ids)}")

    if len(common_ids) > 0:
        # Get indices for common proteins
        idx1 = [data1['protein_ids'].index(pid) for pid in list(common_ids)[:100]]
        idx2 = [data2['protein_ids'].index(pid) for pid in list(common_ids)[:100]]

        common_emb1 = emb1[idx1]
        common_emb2 = emb2[idx2]

        # Compute similarity
        cosine_sim = np.sum(common_emb1 * common_emb2, axis=1) / (
            np.linalg.norm(common_emb1, axis=1) * np.linalg.norm(common_emb2, axis=1)
        )

        print(f"\nCosine similarity (first 100 common proteins):")
        print(f"  Mean: {cosine_sim.mean():.4f}")
        print(f"  Std: {cosine_sim.std():.4f}")
        print(f"  Min: {cosine_sim.min():.4f}")
        print(f"  Max: {cosine_sim.max():.4f}")

        if cosine_sim.mean() > 0.95:
            print("\n✓ Embeddings are very similar (likely same strategy)")
        elif cosine_sim.mean() > 0.8:
            print("\n✓ Embeddings are similar (different strategies but compatible)")
        else:
            print("\n⚠ Embeddings are quite different (check strategies)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test ESM-2 embeddings')
    parser.add_argument('embeddings_path', type=str, help='Path to embeddings pickle file')
    parser.add_argument('--compare', type=str, help='Path to second embeddings file for comparison')

    args = parser.parse_args()

    # Test embeddings
    test_embeddings(args.embeddings_path)

    # Compare if second file provided
    if args.compare:
        print("\n")
        compare_embeddings(args.embeddings_path, args.compare)
