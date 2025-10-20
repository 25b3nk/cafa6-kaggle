"""
PyTorch Dataset classes for CAFA-6 protein function prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle


class ProteinSequenceDataset(Dataset):
    """
    PyTorch Dataset for protein sequences with multi-label GO term annotations
    """

    def __init__(self,
                 protein_ids: List[str],
                 sequences: Dict[str, str],
                 labels: np.ndarray,
                 max_length: int = 1024,
                 tokenizer=None):
        """
        Args:
            protein_ids: List of protein IDs
            sequences: Dictionary mapping protein ID to amino acid sequence
            labels: Multi-label binary matrix (n_proteins, n_go_terms)
            max_length: Maximum sequence length (will truncate longer sequences)
            tokenizer: Optional tokenizer for sequences (e.g., ESM tokenizer)
        """
        self.protein_ids = protein_ids
        self.sequences = sequences
        self.labels = torch.FloatTensor(labels)
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Verify data consistency
        assert len(protein_ids) == len(labels), \
            f"Mismatch: {len(protein_ids)} proteins but {len(labels)} label rows"

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        labels = self.labels[idx]

        # Truncate sequence if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        item = {
            'protein_id': protein_id,
            'sequence': sequence,
            'labels': labels,
            'seq_length': len(sequence)
        }

        # If tokenizer provided, tokenize the sequence
        if self.tokenizer is not None:
            # ESM tokenizer adds special tokens automatically
            encoded = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = encoded['input_ids'].squeeze(0)
            item['attention_mask'] = encoded['attention_mask'].squeeze(0)

        return item


class ProteinEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for pre-computed protein embeddings (faster training)
    """

    def __init__(self,
                 protein_ids: List[str],
                 embeddings: np.ndarray,
                 labels: np.ndarray):
        """
        Args:
            protein_ids: List of protein IDs
            embeddings: Pre-computed embeddings (n_proteins, embedding_dim)
            labels: Multi-label binary matrix (n_proteins, n_go_terms)
        """
        self.protein_ids = protein_ids
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)

        assert len(protein_ids) == len(embeddings) == len(labels), \
            "Mismatch in protein_ids, embeddings, and labels lengths"

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'protein_id': self.protein_ids[idx],
            'embedding': self.embeddings[idx],
            'labels': self.labels[idx]
        }


class MultiAspectProteinDataset(Dataset):
    """
    PyTorch Dataset with separate labels for each aspect (C, F, P)
    Used for multi-task learning
    """

    def __init__(self,
                 protein_ids: List[str],
                 sequences: Dict[str, str],
                 aspect_labels: Dict[str, np.ndarray],
                 max_length: int = 1024,
                 tokenizer=None):
        """
        Args:
            protein_ids: List of protein IDs
            sequences: Dictionary mapping protein ID to amino acid sequence
            aspect_labels: Dict with keys 'C', 'F', 'P', values are label matrices
            max_length: Maximum sequence length
            tokenizer: Optional tokenizer for sequences
        """
        self.protein_ids = protein_ids
        self.sequences = sequences
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Convert aspect labels to tensors
        self.labels_C = torch.FloatTensor(aspect_labels['C'])
        self.labels_F = torch.FloatTensor(aspect_labels['F'])
        self.labels_P = torch.FloatTensor(aspect_labels['P'])

        # Verify consistency
        assert len(protein_ids) == len(self.labels_C) == len(self.labels_F) == len(self.labels_P), \
            "Mismatch in protein_ids and aspect labels lengths"

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]

        # Truncate sequence if needed
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        item = {
            'protein_id': protein_id,
            'sequence': sequence,
            'labels_C': self.labels_C[idx],
            'labels_F': self.labels_F[idx],
            'labels_P': self.labels_P[idx],
            'seq_length': len(sequence)
        }

        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = encoded['input_ids'].squeeze(0)
            item['attention_mask'] = encoded['attention_mask'].squeeze(0)

        return item


class MultiAspectEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for pre-computed embeddings with multi-aspect labels
    """

    def __init__(self,
                 protein_ids: List[str],
                 embeddings: np.ndarray,
                 aspect_labels: Dict[str, np.ndarray]):
        """
        Args:
            protein_ids: List of protein IDs
            embeddings: Pre-computed embeddings (n_proteins, embedding_dim)
            aspect_labels: Dict with keys 'C', 'F', 'P', values are label matrices
        """
        self.protein_ids = protein_ids
        self.embeddings = torch.FloatTensor(embeddings)

        self.labels_C = torch.FloatTensor(aspect_labels['C'])
        self.labels_F = torch.FloatTensor(aspect_labels['F'])
        self.labels_P = torch.FloatTensor(aspect_labels['P'])

        assert len(protein_ids) == len(embeddings), "Mismatch in protein_ids and embeddings"

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'protein_id': self.protein_ids[idx],
            'embedding': self.embeddings[idx],
            'labels_C': self.labels_C[idx],
            'labels_F': self.labels_F[idx],
            'labels_P': self.labels_P[idx]
        }


def create_dataloaders(preprocessed_data_path: str,
                      batch_size: int = 32,
                      val_split: float = 0.1,
                      use_embeddings: bool = False,
                      multi_aspect: bool = False,
                      tokenizer=None,
                      num_workers: int = 4,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from preprocessed data

    Args:
        preprocessed_data_path: Path to preprocessed_data.pkl
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        use_embeddings: If True, use embedding dataset (requires embeddings in data)
        multi_aspect: If True, use multi-aspect dataset
        tokenizer: Tokenizer for sequences (if not using embeddings)
        num_workers: Number of workers for dataloader
        seed: Random seed for train/val split

    Returns:
        train_loader, val_loader
    """
    # Load preprocessed data
    with open(preprocessed_data_path, 'rb') as f:
        data = pickle.load(f)

    protein_ids = data['protein_ids']
    sequences = data['sequences']

    # Create train/val split
    np.random.seed(seed)
    n_proteins = len(protein_ids)
    indices = np.random.permutation(n_proteins)
    val_size = int(n_proteins * val_split)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Split protein IDs
    train_ids = [protein_ids[i] for i in train_indices]
    val_ids = [protein_ids[i] for i in val_indices]

    print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")

    # Create datasets based on configuration
    if multi_aspect:
        # Extract aspect-specific labels
        aspect_labels_data = data['aspect_labels']
        aspect_labels = {
            'C': aspect_labels_data['C'][0],
            'F': aspect_labels_data['F'][0],
            'P': aspect_labels_data['P'][0]
        }

        if use_embeddings:
            embeddings = data['embeddings']  # Assuming embeddings are in data
            train_aspect_labels = {k: v[train_indices] for k, v in aspect_labels.items()}
            val_aspect_labels = {k: v[val_indices] for k, v in aspect_labels.items()}

            train_dataset = MultiAspectEmbeddingDataset(
                train_ids,
                embeddings[train_indices],
                train_aspect_labels
            )
            val_dataset = MultiAspectEmbeddingDataset(
                val_ids,
                embeddings[val_indices],
                val_aspect_labels
            )
        else:
            train_aspect_labels = {k: v[train_indices] for k, v in aspect_labels.items()}
            val_aspect_labels = {k: v[val_indices] for k, v in aspect_labels.items()}

            train_dataset = MultiAspectProteinDataset(
                train_ids,
                sequences,
                train_aspect_labels,
                tokenizer=tokenizer
            )
            val_dataset = MultiAspectProteinDataset(
                val_ids,
                sequences,
                val_aspect_labels,
                tokenizer=tokenizer
            )
    else:
        # Single-task with all labels
        full_labels = data['full_labels']

        if use_embeddings:
            embeddings = data['embeddings']
            train_dataset = ProteinEmbeddingDataset(
                train_ids,
                embeddings[train_indices],
                full_labels[train_indices]
            )
            val_dataset = ProteinEmbeddingDataset(
                val_ids,
                embeddings[val_indices],
                full_labels[val_indices]
            )
        else:
            train_dataset = ProteinSequenceDataset(
                train_ids,
                sequences,
                full_labels[train_indices],
                tokenizer=tokenizer
            )
            val_dataset = ProteinSequenceDataset(
                val_ids,
                sequences,
                full_labels[val_indices],
                tokenizer=tokenizer
            )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Example usage
    from preprocessing import load_preprocessed_data

    # Load preprocessed data
    data = load_preprocessed_data('data/processed/preprocessed_data.pkl')

    # Create a simple dataset
    dataset = ProteinSequenceDataset(
        protein_ids=data['protein_ids'][:100],
        sequences=data['sequences'],
        labels=data['full_labels'][:100]
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sequence length: {sample['seq_length']}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Number of positive labels: {sample['labels'].sum().item()}")
