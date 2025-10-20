"""
Kaggle-optimized ESM-2 embedding generation
Handles memory constraints and saves embeddings in chunks

This script is optimized for Kaggle's GPU environment:
- P100 GPU: 16GB RAM
- T4 GPU: 16GB RAM
- Processes in smaller batches
- Saves embeddings in chunks to avoid memory issues
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import gc


def generate_embeddings_kaggle(data_dir: str,
                               output_dir: str = '/kaggle/working/embeddings',
                               model_name: str = 'esm2_t33_650M_UR50D',
                               split: str = 'train',
                               batch_size: int = 4,
                               max_length: int = 1024,
                               strategy: str = 'balanced',
                               chunk_size: int = 5000):
    """
    Generate ESM-2 embeddings optimized for Kaggle environment

    Args:
        data_dir: Path to data directory
        output_dir: Where to save embeddings
        model_name: ESM-2 model variant
        split: 'train' or 'test'
        batch_size: Smaller batch size for Kaggle (4-8 recommended)
        max_length: Max sequence length
        strategy: Truncation strategy
        chunk_size: Save embeddings every N proteins (memory management)
    """

    # Add src to path
    import sys
    sys.path.append('..')

    from src.data import ProteinDataPreprocessor
    from src.utils import LongSequenceHandler

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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
    model = model.to(device)
    model.eval()

    embedding_dim = model.embed_dim
    num_layers = model.num_layers
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

    print(f"\nGenerating embeddings (batch_size={batch_size}, chunk_size={chunk_size})...")

    for i in tqdm(range(0, len(protein_ids), batch_size)):
        batch_ids = protein_ids[i:i + batch_size]
        batch_data = [(pid, truncate_seq(sequences[pid])) for pid in batch_ids]

        # Convert batch
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Generate embeddings (use last layer)
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
            if device == 'cuda':
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

    for chunk_num in range(chunk_num + 1):
        chunk_path = output_path / f'embeddings_chunk_{chunk_num}.pkl'
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
        'num_proteins': len(all_protein_ids)
    }

    with open(final_path, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"\n{'='*60}")
    print(f"Embeddings saved to: {final_path}")
    print(f"Shape: {embeddings_array.shape}")
    print(f"Size: {final_path.stat().st_size / (1024**3):.2f} GB")
    print(f"{'='*60}")

    return final_path


# Example usage in Kaggle notebook:
"""
# In Kaggle notebook cell:

!pip install fair-esm

# Generate embeddings
embeddings_path = generate_embeddings_kaggle(
    data_dir='/kaggle/input/cafa-6-protein-function-prediction',
    output_dir='/kaggle/working/embeddings',
    model_name='esm2_t33_650M_UR50D',
    split='train',
    batch_size=4,  # Small batch for memory
    strategy='balanced'
)

# Load embeddings for training
import pickle
with open(embeddings_path, 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']  # numpy array
protein_ids = data['protein_ids']

print(f"Embeddings shape: {embeddings.shape}")
"""
