"""
Memory-efficient submission file creation for CAFA-6
Processes predictions in batches and writes incrementally to avoid RAM overload

Usage:
    python scripts/create_submission.py \
        --model models/baseline/best_model.pt \
        --test_embeddings data/embeddings/test_embeddings.pkl \
        --output submissions/submission.csv \
        --threshold 0.5 \
        --batch_size 32
"""

import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import SimpleProteinClassifier


def create_submission_incremental(model,
                                   test_embeddings,
                                   protein_ids,
                                   idx_to_go_term,
                                   output_path,
                                   threshold=0.5,
                                   batch_size=32,
                                   min_predictions_per_protein=1):
    """
    Create submission file incrementally to avoid RAM overload

    Args:
        model: Trained model
        test_embeddings: (num_proteins, embedding_dim) array
        protein_ids: List of protein IDs
        idx_to_go_term: Dictionary mapping index to GO term
        output_path: Path to save submission CSV
        threshold: Prediction threshold
        batch_size: Batch size for inference
        min_predictions_per_protein: Minimum predictions per protein (CAFA requirement)
    """

    print(f"Creating submission for {len(protein_ids)} proteins...")
    print(f"Batch size: {batch_size}")
    print(f"Threshold: {threshold}")

    model.eval()
    device = next(model.parameters()).device

    # Open file for writing (append mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header (TSV format - tab separated)
    with open(output_path, 'w') as f:
        f.write("Protein ID\tGO Term\tConfidence\n")

    total_predictions = 0
    proteins_with_no_predictions = 0

    # Process in batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(protein_ids), batch_size), desc="Generating predictions"):
            end_idx = min(start_idx + batch_size, len(protein_ids))

            # Get batch
            batch_embeddings = test_embeddings[start_idx:end_idx]
            batch_protein_ids = protein_ids[start_idx:end_idx]

            # Predict
            batch_tensor = torch.FloatTensor(batch_embeddings).to(device)
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Process each protein in batch
            batch_rows = []
            for i, protein_id in enumerate(batch_protein_ids):
                protein_probs = probs[i]

                # Get predictions above threshold
                pred_indices = np.where(protein_probs > threshold)[0]

                # If no predictions, use top prediction(s)
                if len(pred_indices) == 0:
                    proteins_with_no_predictions += 1
                    # Get top N predictions
                    top_n = min(min_predictions_per_protein, len(protein_probs))
                    pred_indices = np.argsort(protein_probs)[-top_n:]

                # Create rows for this protein (TSV format - tab separated)
                for idx in pred_indices:
                    go_term = idx_to_go_term[idx]
                    confidence = float(protein_probs[idx])
                    batch_rows.append(f"{protein_id}\t{go_term}\t{confidence:.6f}\n")
                    total_predictions += 1

            # Write batch to file
            with open(output_path, 'a') as f:
                f.writelines(batch_rows)

            # Clear memory
            del batch_tensor, outputs, probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Submission created: {output_path}")
    print(f"Total predictions: {total_predictions}")
    print(f"Proteins with no predictions above threshold: {proteins_with_no_predictions}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Create CAFA-6 submission file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_embeddings', type=str, required=True,
                        help='Path to test embeddings pickle file')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to preprocessed labels (for GO term mappings)')
    parser.add_argument('--output', type=str, default='submissions/submission.tsv',
                        help='Output submission file path (TSV format)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    print("="*60)
    print("CAFA-6 Submission File Creation")
    print("="*60)

    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load test embeddings
    print(f"\nLoading test embeddings from {args.test_embeddings}...")
    with open(args.test_embeddings, 'rb') as f:
        test_data = pickle.load(f)

    test_embeddings = test_data['embeddings']
    protein_ids = test_data['protein_ids']
    print(f"Loaded {len(protein_ids)} test proteins")
    print(f"Embedding shape: {test_embeddings.shape}")

    # Load labels (for GO term mappings)
    print(f"\nLoading label mappings from {args.labels}...")
    with open(args.labels, 'rb') as f:
        label_data = pickle.load(f)

    idx_to_go_term = label_data['idx_to_go_term']
    num_classes = len(idx_to_go_term)
    print(f"Number of GO terms: {num_classes}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)

    # Reconstruct model
    embedding_dim = checkpoint.get('embedding_dim', test_embeddings.shape[1])

    model = SimpleProteinClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Output classes: {num_classes}")

    # Create submission
    create_submission_incremental(
        model=model,
        test_embeddings=test_embeddings,
        protein_ids=protein_ids,
        idx_to_go_term=idx_to_go_term,
        output_path=args.output,
        threshold=args.threshold,
        batch_size=args.batch_size
    )

    # Validate submission format
    print("\nValidating submission format...")
    df = pd.read_csv(args.output, sep='\t', nrows=10)
    print(f"First 10 rows:")
    print(df)

    # Check required columns
    required_cols = ['Protein ID', 'GO Term', 'Confidence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print(f"✓ All required columns present")

    print("\n✓ Submission file created successfully!")


if __name__ == '__main__':
    main()
