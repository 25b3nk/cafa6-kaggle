"""
Generate ESM-2 embeddings for all protein sequences
Handles long sequences using balanced truncation or sliding window

Usage:
    python scripts/generate_embeddings.py --data_dir cafa-6-protein-function-prediction \
                                          --output_dir data/embeddings \
                                          --model_name esm2_t33_650M_UR50D \
                                          --strategy balanced \
                                          --batch_size 8
"""

import torch
import numpy as np
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import ProteinDataPreprocessor
from src.utils import LongSequenceHandler


class ESM2EmbeddingGenerator:
    """
    Generate ESM-2 embeddings for protein sequences
    """

    def __init__(self,
                 model_name: str = 'esm2_t33_650M_UR50D',
                 device: str = 'cuda',
                 max_length: int = 1024,
                 strategy: str = 'balanced'):
        """
        Args:
            model_name: ESM-2 model name
            device: 'cuda' or 'cpu'
            max_length: Maximum sequence length
            strategy: 'balanced', 'sliding_window', or 'start'
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_length = max_length
        self.strategy = strategy
        self.handler = LongSequenceHandler()

        print(f"Initializing ESM-2 model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Strategy: {strategy}")
        print(f"Max length: {max_length}")

        # Load ESM-2 model
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.embed_dim
            print(f"Embedding dimension: {self.embedding_dim}")

        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM-2 model: {e}")

    def truncate_sequence(self, sequence: str) -> str:
        """
        Truncate sequence based on strategy

        Args:
            sequence: Amino acid sequence

        Returns:
            Truncated sequence
        """
        if len(sequence) <= self.max_length:
            return sequence

        if self.strategy == 'balanced':
            return self.handler.truncate_balanced(sequence, self.max_length)
        elif self.strategy == 'start':
            return self.handler.truncate_start(sequence, self.max_length)
        elif self.strategy == 'end':
            return self.handler.truncate_end(sequence, self.max_length)
        elif self.strategy == 'middle':
            return self.handler.truncate_middle(sequence, self.max_length)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def embed_batch(self, batch_data: list) -> torch.Tensor:
        """
        Generate embeddings for a batch of sequences

        Args:
            batch_data: List of (protein_id, sequence) tuples

        Returns:
            Tensor of embeddings (batch_size, embedding_dim)
        """
        # Truncate long sequences
        truncated_data = [
            (pid, self.truncate_sequence(seq))
            for pid, seq in batch_data
        ]

        # Convert batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(truncated_data)
        batch_tokens = batch_tokens.to(self.device)

        # Generate embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            embeddings = results['representations'][33]

        # Extract [CLS] token embeddings (position 0)
        cls_embeddings = embeddings[:, 0, :]

        return cls_embeddings.cpu()

    def generate_embeddings(self,
                          sequences: dict,
                          batch_size: int = 8,
                          save_path: str = None) -> dict:
        """
        Generate embeddings for all sequences

        Args:
            sequences: Dictionary mapping protein_id to sequence
            batch_size: Batch size for processing
            save_path: Path to save embeddings (optional)

        Returns:
            Dictionary mapping protein_id to embedding
        """
        protein_ids = list(sequences.keys())
        num_proteins = len(protein_ids)

        print(f"\nGenerating embeddings for {num_proteins} proteins...")
        print(f"Batch size: {batch_size}")

        embeddings_dict = {}
        embeddings_list = []
        protein_id_list = []

        # Track statistics
        num_truncated = 0
        total_aa_lost = 0

        # Process in batches
        for i in tqdm(range(0, num_proteins, batch_size), desc="Generating embeddings"):
            batch_ids = protein_ids[i:i + batch_size]
            batch_data = [(pid, sequences[pid]) for pid in batch_ids]

            # Track truncation
            for pid, seq in batch_data:
                if len(seq) > self.max_length:
                    num_truncated += 1
                    total_aa_lost += len(seq) - self.max_length

            # Generate embeddings
            batch_embeddings = self.embed_batch(batch_data)

            # Store embeddings
            for pid, emb in zip(batch_ids, batch_embeddings):
                embeddings_dict[pid] = emb.numpy()
                embeddings_list.append(emb.numpy())
                protein_id_list.append(pid)

        # Convert to numpy array
        embeddings_array = np.array(embeddings_list)

        print(f"\nEmbedding generation complete!")
        print(f"Total proteins: {num_proteins}")
        print(f"Truncated sequences: {num_truncated} ({num_truncated/num_proteins*100:.1f}%)")
        if num_truncated > 0:
            print(f"Average AA lost per truncated sequence: {total_aa_lost/num_truncated:.1f}")
        print(f"Embedding shape: {embeddings_array.shape}")

        # Save embeddings if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            embeddings_data = {
                'embeddings': embeddings_array,
                'protein_ids': protein_id_list,
                'embeddings_dict': embeddings_dict,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'max_length': self.max_length,
                'strategy': self.strategy,
                'num_truncated': num_truncated,
                'num_proteins': num_proteins
            }

            with open(save_path, 'wb') as f:
                pickle.dump(embeddings_data, f)

            print(f"\nEmbeddings saved to: {save_path}")
            print(f"File size: {save_path.stat().st_size / (1024**3):.2f} GB")

        return embeddings_dict


def load_embeddings(filepath: str) -> dict:
    """
    Load pre-computed embeddings from file

    Args:
        filepath: Path to embeddings pickle file

    Returns:
        Dictionary with embeddings and metadata
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded embeddings from: {filepath}")
    print(f"Model: {data['model_name']}")
    print(f"Embedding dim: {data['embedding_dim']}")
    print(f"Num proteins: {data['num_proteins']}")
    print(f"Truncated: {data['num_truncated']} ({data['num_truncated']/data['num_proteins']*100:.1f}%)")

    return data


def main():
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings for CAFA-6 proteins')

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to cafa-6-protein-function-prediction directory')
    parser.add_argument('--output_dir', type=str, default='data/embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Which split to process')
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM-2 model name')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for embedding generation')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--strategy', type=str, default='balanced',
                       choices=['balanced', 'start', 'end', 'middle', 'sliding_window'],
                       help='Truncation strategy for long sequences')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU instead")
        args.device = 'cpu'

    # Load sequences
    print(f"\n{'='*60}")
    print("Loading protein sequences...")
    print(f"{'='*60}")

    preprocessor = ProteinDataPreprocessor(args.data_dir)
    sequences = preprocessor.load_sequences(args.split)

    # Analyze sequence lengths
    lengths = [len(seq) for seq in sequences.values()]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  > {args.max_length}: {sum(l > args.max_length for l in lengths)} "
          f"({sum(l > args.max_length for l in lengths)/len(lengths)*100:.1f}%)")

    # Initialize generator
    print(f"\n{'='*60}")
    print("Initializing ESM-2 model...")
    print(f"{'='*60}")

    generator = ESM2EmbeddingGenerator(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        strategy=args.strategy
    )

    # Generate embeddings
    print(f"\n{'='*60}")
    print("Generating embeddings...")
    print(f"{'='*60}")

    output_path = Path(args.output_dir) / f"{args.split}_embeddings_{args.model_name}_{args.strategy}.pkl"

    embeddings = generator.generate_embeddings(
        sequences,
        batch_size=args.batch_size,
        save_path=output_path
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
