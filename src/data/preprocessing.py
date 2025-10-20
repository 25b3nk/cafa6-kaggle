"""
Data preprocessing utilities for CAFA-6 protein function prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle


class ProteinDataPreprocessor:
    """
    Preprocesses protein sequences and GO term annotations for model training
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to cafa-6-protein-function-prediction directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'Train'
        self.test_dir = self.data_dir / 'Test'

        # Will be populated during preprocessing
        self.go_term_to_idx = {}
        self.idx_to_go_term = {}
        self.go_term_to_aspect = {}
        self.aspect_to_terms = {'C': [], 'F': [], 'P': []}
        self.protein_to_sequence = {}
        self.protein_to_labels = {}
        self.class_weights = None

    def load_sequences(self, split='train') -> Dict[str, str]:
        """
        Load protein sequences from FASTA file

        Args:
            split: 'train' or 'test'

        Returns:
            Dictionary mapping protein ID to amino acid sequence
        """
        if split == 'train':
            fasta_file = self.train_dir / 'train_sequences.fasta'
        else:
            fasta_file = self.test_dir / 'testsuperset.fasta'

        sequences = {}
        current_id = None
        current_seq = []

        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    # Start new sequence
                    # Format: >sp|Q5W0B1|OBI1_HUMAN ...
                    current_id = line[1:].split()[0]
                    # Extract just the UniProt ID (Q5W0B1)
                    if '|' in current_id:
                        current_id = current_id.split('|')[1]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last sequence
            if current_id:
                sequences[current_id] = ''.join(current_seq)

        print(f"Loaded {len(sequences)} {split} sequences")
        return sequences

    def load_annotations(self) -> pd.DataFrame:
        """
        Load GO term annotations from train_terms.tsv

        Returns:
            DataFrame with columns: EntryID, term, aspect
        """
        annotations_file = self.train_dir / 'train_terms.tsv'
        df = pd.read_csv(annotations_file, sep='\t')
        print(f"Loaded {len(df)} annotations for {df['EntryID'].nunique()} proteins")
        return df

    def build_label_encodings(self, annotations_df: pd.DataFrame):
        """
        Build mappings between GO terms and integer indices

        Args:
            annotations_df: DataFrame from load_annotations()
        """
        # Get unique GO terms
        unique_terms = sorted(annotations_df['term'].unique())

        # Create term to index mapping
        self.go_term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
        self.idx_to_go_term = {idx: term for term, idx in self.go_term_to_idx.items()}

        # Create term to aspect mapping
        term_aspect_map = annotations_df[['term', 'aspect']].drop_duplicates()
        self.go_term_to_aspect = dict(zip(term_aspect_map['term'], term_aspect_map['aspect']))

        # Group terms by aspect
        for term, aspect in self.go_term_to_aspect.items():
            self.aspect_to_terms[aspect].append(term)

        print(f"Total GO terms: {len(self.go_term_to_idx)}")
        print(f"  C (Component): {len(self.aspect_to_terms['C'])} terms")
        print(f"  F (Function): {len(self.aspect_to_terms['F'])} terms")
        print(f"  P (Process): {len(self.aspect_to_terms['P'])} terms")

    def compute_class_weights(self, annotations_df: pd.DataFrame,
                             method='balanced') -> np.ndarray:
        """
        Compute class weights to handle imbalance

        Args:
            annotations_df: DataFrame from load_annotations()
            method: 'balanced' or 'log' or 'sqrt'

        Returns:
            Array of weights for each GO term
        """
        # Count frequency of each GO term
        term_counts = annotations_df['term'].value_counts()

        # Initialize weights array
        weights = np.ones(len(self.go_term_to_idx))

        total_samples = len(annotations_df)
        n_classes = len(self.go_term_to_idx)

        for term, count in term_counts.items():
            idx = self.go_term_to_idx[term]

            if method == 'balanced':
                # Sklearn-style balanced weight
                weights[idx] = total_samples / (n_classes * count)
            elif method == 'log':
                # Logarithmic scaling
                weights[idx] = np.log(total_samples / count + 1)
            elif method == 'sqrt':
                # Square root scaling (less aggressive)
                weights[idx] = np.sqrt(total_samples / count)

        # Normalize weights
        weights = weights / weights.mean()

        self.class_weights = weights
        print(f"Computed class weights (method={method})")
        print(f"  Min weight: {weights.min():.4f}")
        print(f"  Max weight: {weights.max():.4f}")
        print(f"  Mean weight: {weights.mean():.4f}")

        return weights

    def create_multi_label_matrix(self, annotations_df: pd.DataFrame,
                                  protein_ids: List[str]) -> np.ndarray:
        """
        Create multi-label binary matrix for training

        Args:
            annotations_df: DataFrame from load_annotations()
            protein_ids: List of protein IDs in order

        Returns:
            Binary matrix of shape (n_proteins, n_go_terms)
        """
        n_proteins = len(protein_ids)
        n_terms = len(self.go_term_to_idx)

        # Initialize label matrix
        labels = np.zeros((n_proteins, n_terms), dtype=np.float32)

        # Create protein ID to index mapping
        protein_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}

        # Fill in labels
        for _, row in annotations_df.iterrows():
            protein_id = row['EntryID']
            term = row['term']

            if protein_id in protein_to_idx and term in self.go_term_to_idx:
                protein_idx = protein_to_idx[protein_id]
                term_idx = self.go_term_to_idx[term]
                labels[protein_idx, term_idx] = 1.0

        # Print statistics
        n_annotations = labels.sum()
        avg_labels_per_protein = labels.sum(axis=1).mean()
        avg_proteins_per_label = labels.sum(axis=0).mean()

        print(f"Created label matrix: {labels.shape}")
        print(f"  Total annotations: {int(n_annotations)}")
        print(f"  Avg labels per protein: {avg_labels_per_protein:.2f}")
        print(f"  Avg proteins per label: {avg_proteins_per_label:.2f}")
        print(f"  Sparsity: {(1 - n_annotations / labels.size) * 100:.2f}%")

        return labels

    def create_aspect_specific_matrices(self, annotations_df: pd.DataFrame,
                                       protein_ids: List[str]) -> Dict[str, Tuple[np.ndarray, List[int]]]:
        """
        Create separate label matrices for each aspect (C, F, P)

        Args:
            annotations_df: DataFrame from load_annotations()
            protein_ids: List of protein IDs in order

        Returns:
            Dictionary with keys 'C', 'F', 'P', each containing:
                - Binary label matrix for that aspect
                - List of GO term indices for that aspect
        """
        aspect_data = {}

        for aspect in ['C', 'F', 'P']:
            # Get GO term indices for this aspect
            aspect_terms = self.aspect_to_terms[aspect]
            aspect_term_indices = [self.go_term_to_idx[term] for term in aspect_terms]

            # Create mapping for aspect-specific indexing
            aspect_idx_map = {global_idx: local_idx
                            for local_idx, global_idx in enumerate(aspect_term_indices)}

            # Initialize label matrix
            n_proteins = len(protein_ids)
            n_aspect_terms = len(aspect_term_indices)
            labels = np.zeros((n_proteins, n_aspect_terms), dtype=np.float32)

            # Create protein ID to index mapping
            protein_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}

            # Filter annotations for this aspect
            aspect_annotations = annotations_df[annotations_df['aspect'] == aspect]

            # Fill in labels
            for _, row in aspect_annotations.iterrows():
                protein_id = row['EntryID']
                term = row['term']

                if protein_id in protein_to_idx and term in self.go_term_to_idx:
                    protein_idx = protein_to_idx[protein_id]
                    global_term_idx = self.go_term_to_idx[term]
                    local_term_idx = aspect_idx_map[global_term_idx]
                    labels[protein_idx, local_term_idx] = 1.0

            aspect_data[aspect] = (labels, aspect_term_indices)

            print(f"Aspect {aspect}: {labels.shape}, {int(labels.sum())} annotations")

        return aspect_data

    def preprocess_all(self, output_dir: Optional[str] = None):
        """
        Run full preprocessing pipeline and save results

        Args:
            output_dir: Directory to save preprocessed data (optional)
        """
        print("="*60)
        print("Starting preprocessing pipeline...")
        print("="*60)

        # Load sequences
        self.protein_to_sequence = self.load_sequences('train')

        # Load annotations
        annotations_df = self.load_annotations()

        # Build label encodings
        self.build_label_encodings(annotations_df)

        # Get protein IDs (only those with both sequence and annotations)
        protein_ids = sorted(list(set(self.protein_to_sequence.keys()) &
                                 set(annotations_df['EntryID'].unique())))
        print(f"\nProteins with both sequence and annotations: {len(protein_ids)}")

        # Create label matrices
        print("\nCreating label matrices...")
        full_labels = self.create_multi_label_matrix(annotations_df, protein_ids)
        aspect_labels = self.create_aspect_specific_matrices(annotations_df, protein_ids)

        # Compute class weights
        print("\nComputing class weights...")
        class_weights = self.compute_class_weights(annotations_df, method='balanced')

        # Save preprocessed data if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            preprocessed_data = {
                'protein_ids': protein_ids,
                'sequences': {pid: self.protein_to_sequence[pid] for pid in protein_ids},
                'full_labels': full_labels,
                'aspect_labels': aspect_labels,
                'go_term_to_idx': self.go_term_to_idx,
                'idx_to_go_term': self.idx_to_go_term,
                'go_term_to_aspect': self.go_term_to_aspect,
                'aspect_to_terms': self.aspect_to_terms,
                'class_weights': class_weights
            }

            save_path = output_path / 'preprocessed_data.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(preprocessed_data, f)

            print(f"\nPreprocessed data saved to {save_path}")

        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60)

        return {
            'protein_ids': protein_ids,
            'full_labels': full_labels,
            'aspect_labels': aspect_labels,
            'class_weights': class_weights
        }


def load_preprocessed_data(filepath: str) -> Dict:
    """
    Load preprocessed data from pickle file

    Args:
        filepath: Path to preprocessed_data.pkl

    Returns:
        Dictionary with preprocessed data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded preprocessed data with {len(data['protein_ids'])} proteins")
    return data


if __name__ == '__main__':
    # Example usage
    preprocessor = ProteinDataPreprocessor('cafa-6-protein-function-prediction')
    preprocessed_data = preprocessor.preprocess_all(output_dir='data/processed')
