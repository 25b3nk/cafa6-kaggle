"""
Utilities for handling long protein sequences
Strategies to preserve context in sequences longer than model max length
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


class LongSequenceHandler:
    """
    Strategies for handling protein sequences longer than model max length
    """

    @staticmethod
    def truncate_start(sequence: str, max_length: int) -> str:
        """
        Keep first max_length amino acids (N-terminus)

        Args:
            sequence: Amino acid sequence
            max_length: Maximum length

        Returns:
            Truncated sequence
        """
        return sequence[:max_length]

    @staticmethod
    def truncate_end(sequence: str, max_length: int) -> str:
        """
        Keep last max_length amino acids (C-terminus)

        Args:
            sequence: Amino acid sequence
            max_length: Maximum length

        Returns:
            Truncated sequence
        """
        return sequence[-max_length:]

    @staticmethod
    def truncate_middle(sequence: str, max_length: int) -> str:
        """
        Keep middle max_length amino acids

        Args:
            sequence: Amino acid sequence
            max_length: Maximum length

        Returns:
            Truncated sequence
        """
        if len(sequence) <= max_length:
            return sequence

        start_idx = (len(sequence) - max_length) // 2
        return sequence[start_idx:start_idx + max_length]

    @staticmethod
    def truncate_balanced(sequence: str, max_length: int,
                         n_term_ratio: float = 0.5) -> str:
        """
        Keep both N-terminus and C-terminus (most common strategy)

        Args:
            sequence: Amino acid sequence
            max_length: Maximum length
            n_term_ratio: Fraction to allocate to N-terminus (default 0.5 = equal)

        Returns:
            Truncated sequence with both termini

        Example:
            If sequence is 2000 AA and max_length is 1024:
            - Keep first 512 AA (N-terminus)
            - Keep last 512 AA (C-terminus)
            - Concatenate: "FIRST512..." + "...LAST512"
        """
        if len(sequence) <= max_length:
            return sequence

        n_term_length = int(max_length * n_term_ratio)
        c_term_length = max_length - n_term_length

        return sequence[:n_term_length] + sequence[-c_term_length:]

    @staticmethod
    def sliding_window_chunks(sequence: str,
                             max_length: int,
                             stride: int = None,
                             overlap: int = 128) -> List[str]:
        """
        Split long sequence into overlapping chunks

        Args:
            sequence: Amino acid sequence
            max_length: Maximum chunk length
            stride: Step size between chunks (if None, uses max_length - overlap)
            overlap: Number of amino acids to overlap between chunks

        Returns:
            List of sequence chunks

        Example:
            sequence of 2000 AA, max_length=1024, overlap=128
            Returns: [chunk1(0-1024), chunk2(896-1920), chunk3(1792-2000)]
        """
        if len(sequence) <= max_length:
            return [sequence]

        if stride is None:
            stride = max_length - overlap

        chunks = []
        start = 0

        while start < len(sequence):
            end = min(start + max_length, len(sequence))
            chunks.append(sequence[start:end])

            if end == len(sequence):
                break

            start += stride

        return chunks

    @staticmethod
    def hierarchical_chunks(sequence: str,
                           max_length: int,
                           num_chunks: int = 3) -> List[str]:
        """
        Split sequence into fixed number of non-overlapping chunks
        Useful for hierarchical models

        Args:
            sequence: Amino acid sequence
            max_length: Maximum chunk length
            num_chunks: Number of chunks to create

        Returns:
            List of sequence chunks (may be padded to max_length)
        """
        if len(sequence) <= max_length:
            return [sequence]

        chunk_size = len(sequence) // num_chunks
        chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(sequence)
            chunk = sequence[start:end]

            # Truncate if chunk is too long
            if len(chunk) > max_length:
                chunk = chunk[:max_length]

            chunks.append(chunk)

        return chunks


class MultiChunkEmbedder:
    """
    Generate embeddings for long sequences by processing multiple chunks
    """

    def __init__(self, model, tokenizer, max_length: int = 1024,
                 strategy: str = 'sliding_window'):
        """
        Args:
            model: Protein language model (ESM-2, ProtBERT, etc.)
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length model can handle
            strategy: 'sliding_window', 'balanced', 'hierarchical'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strategy = strategy
        self.handler = LongSequenceHandler()

    def embed_sequence(self, sequence: str,
                      pooling: str = 'mean') -> torch.Tensor:
        """
        Generate embedding for a potentially long sequence

        Args:
            sequence: Amino acid sequence
            pooling: How to combine chunk embeddings ('mean', 'max', 'concat', 'cls')

        Returns:
            Embedding tensor
        """
        if len(sequence) <= self.max_length:
            # Sequence fits in context, process normally
            return self._embed_single_chunk(sequence)

        # Split into chunks based on strategy
        if self.strategy == 'sliding_window':
            chunks = self.handler.sliding_window_chunks(sequence, self.max_length)
        elif self.strategy == 'balanced':
            # For balanced, just use one balanced chunk
            chunks = [self.handler.truncate_balanced(sequence, self.max_length)]
        elif self.strategy == 'hierarchical':
            chunks = self.handler.hierarchical_chunks(sequence, self.max_length, num_chunks=3)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Get embeddings for all chunks
        chunk_embeddings = []
        for chunk in chunks:
            emb = self._embed_single_chunk(chunk)
            chunk_embeddings.append(emb)

        # Combine chunk embeddings
        chunk_embeddings = torch.stack(chunk_embeddings, dim=0)  # (num_chunks, embed_dim)

        if pooling == 'mean':
            return chunk_embeddings.mean(dim=0)
        elif pooling == 'max':
            return chunk_embeddings.max(dim=0)[0]
        elif pooling == 'concat':
            return chunk_embeddings.flatten()
        elif pooling == 'cls':
            # Use only the first chunk's embedding (similar to BERT)
            return chunk_embeddings[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def _embed_single_chunk(self, sequence: str) -> torch.Tensor:
        """
        Generate embedding for a single sequence chunk

        Args:
            sequence: Amino acid sequence (must be <= max_length)

        Returns:
            Embedding tensor
        """
        # Tokenize
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get model embeddings
        with torch.no_grad():
            if hasattr(self.model, 'esm_model'):  # ESM-2
                results = self.model.esm_model(
                    encoded['input_ids'],
                    repr_layers=[33]
                )
                embedding = results['representations'][33][:, 0, :]  # CLS token
            else:  # BERT-style
                outputs = self.model.encoder(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
                embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

        return embedding.squeeze(0)


class SequenceAugmentation:
    """
    Data augmentation for protein sequences
    Helps model learn invariances and improves generalization
    """

    @staticmethod
    def random_crop(sequence: str, crop_length: int) -> str:
        """
        Randomly crop a subsequence

        Args:
            sequence: Original sequence
            crop_length: Length of crop

        Returns:
            Cropped sequence
        """
        if len(sequence) <= crop_length:
            return sequence

        max_start = len(sequence) - crop_length
        start = np.random.randint(0, max_start)
        return sequence[start:start + crop_length]

    @staticmethod
    def random_truncate(sequence: str, min_ratio: float = 0.8) -> str:
        """
        Randomly truncate sequence to simulate incomplete data

        Args:
            sequence: Original sequence
            min_ratio: Minimum fraction to keep (0.8 = keep at least 80%)

        Returns:
            Truncated sequence
        """
        min_length = int(len(sequence) * min_ratio)
        new_length = np.random.randint(min_length, len(sequence) + 1)
        return sequence[:new_length]

    @staticmethod
    def flip_termini(sequence: str) -> str:
        """
        Swap N-terminus and C-terminus
        Tests if model relies too heavily on sequence order

        Args:
            sequence: Original sequence

        Returns:
            Flipped sequence
        """
        return sequence[::-1]


def get_optimal_truncation_strategy(sequence_lengths: List[int],
                                   max_length: int = 1024) -> Dict[str, float]:
    """
    Analyze sequence length distribution and recommend truncation strategy

    Args:
        sequence_lengths: List of all sequence lengths in dataset
        max_length: Model's maximum context length

    Returns:
        Dictionary with statistics and recommendations
    """
    lengths = np.array(sequence_lengths)

    pct_truncated = (lengths > max_length).mean() * 100
    pct_lost = np.where(lengths > max_length,
                        (lengths - max_length) / lengths,
                        0).mean() * 100

    avg_length = lengths.mean()
    median_length = np.median(lengths)
    p95_length = np.percentile(lengths, 95)

    # Recommendations
    if pct_truncated < 10:
        strategy = "truncate_start"
        reason = "< 10% of sequences truncated, simple truncation is fine"
    elif pct_truncated < 30:
        strategy = "truncate_balanced"
        reason = "10-30% truncated, keep both termini"
    else:
        strategy = "sliding_window"
        reason = "> 30% truncated, use sliding window with pooling"

    return {
        'pct_truncated': pct_truncated,
        'pct_info_lost': pct_lost,
        'avg_length': avg_length,
        'median_length': median_length,
        'p95_length': p95_length,
        'recommended_strategy': strategy,
        'reason': reason,
        'max_length': max_length
    }


if __name__ == '__main__':
    # Example usage
    handler = LongSequenceHandler()

    # Long sequence (2000 AA)
    long_seq = "A" * 2000
    max_len = 1024

    print("Testing truncation strategies...")
    print(f"Original length: {len(long_seq)}")
    print(f"Max length: {max_len}\n")

    # Test balanced truncation
    balanced = handler.truncate_balanced(long_seq, max_len)
    print(f"Balanced truncation: {len(balanced)} AA")
    print(f"  Keeps first 512 + last 512\n")

    # Test sliding window
    chunks = handler.sliding_window_chunks(long_seq, max_len, overlap=128)
    print(f"Sliding window: {len(chunks)} chunks")
    print(f"  Chunk lengths: {[len(c) for c in chunks]}\n")

    # Analyze dataset
    print("Dataset analysis...")
    sequence_lengths = [100, 250, 500, 800, 1200, 1500, 2000, 5000, 10000]
    analysis = get_optimal_truncation_strategy(sequence_lengths, max_len)

    print(f"Sequences > max_length: {analysis['pct_truncated']:.1f}%")
    print(f"Average info lost: {analysis['pct_info_lost']:.1f}%")
    print(f"Recommended strategy: {analysis['recommended_strategy']}")
    print(f"Reason: {analysis['reason']}")
