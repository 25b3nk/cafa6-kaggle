"""
Evaluation script for CAFA-6 models
Calculates comprehensive metrics and generates submission files

Usage:
    python scripts/evaluate.py \
        --model models/baseline/best_model.pt \
        --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
        --labels data/processed/preprocessed_data.pkl \
        --output_dir results/baseline
"""

import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import SimpleProteinClassifier, MultiTaskProteinClassifier
from src.data import ProteinEmbeddingDataset, MultiAspectEmbeddingDataset


def evaluate_model(model, dataloader, device='cuda', multitask=False):
    """
    Evaluate model and return predictions

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to use
        multitask: Whether this is a multi-task model

    Returns:
        predictions: Dictionary with predictions and labels
    """
    model.eval()

    all_protein_ids = []
    all_predictions = []
    all_labels = []

    if multitask:
        all_predictions_C = []
        all_predictions_F = []
        all_predictions_P = []
        all_labels_C = []
        all_labels_F = []
        all_labels_P = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            protein_ids = batch['protein_id']
            embeddings = batch['embedding'].to(device)

            if multitask:
                labels_C = batch['labels_C'].numpy()
                labels_F = batch['labels_F'].numpy()
                labels_P = batch['labels_P'].numpy()

                outputs = model(embeddings)

                preds_C = torch.sigmoid(outputs['C']).cpu().numpy()
                preds_F = torch.sigmoid(outputs['F']).cpu().numpy()
                preds_P = torch.sigmoid(outputs['P']).cpu().numpy()

                all_predictions_C.append(preds_C)
                all_predictions_F.append(preds_F)
                all_predictions_P.append(preds_P)

                all_labels_C.append(labels_C)
                all_labels_F.append(labels_F)
                all_labels_P.append(labels_P)
            else:
                labels = batch['labels'].numpy()
                outputs = model(embeddings)
                preds = torch.sigmoid(outputs).cpu().numpy()

                all_predictions.append(preds)
                all_labels.append(labels)

            all_protein_ids.extend(protein_ids)

    if multitask:
        return {
            'protein_ids': all_protein_ids,
            'predictions_C': np.vstack(all_predictions_C),
            'predictions_F': np.vstack(all_predictions_F),
            'predictions_P': np.vstack(all_predictions_P),
            'labels_C': np.vstack(all_labels_C),
            'labels_F': np.vstack(all_labels_F),
            'labels_P': np.vstack(all_labels_P)
        }
    else:
        return {
            'protein_ids': all_protein_ids,
            'predictions': np.vstack(all_predictions),
            'labels': np.vstack(all_labels)
        }


def calculate_metrics(predictions, labels, threshold=0.5):
    """
    Calculate comprehensive metrics

    Args:
        predictions: Predicted probabilities (n_samples, n_classes)
        labels: Ground truth labels (n_samples, n_classes)
        threshold: Threshold for binary predictions

    Returns:
        Dictionary with metrics
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).astype(int)

    # Calculate metrics
    metrics = {
        'f1_micro': f1_score(labels, pred_binary, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, pred_binary, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, pred_binary, average='weighted', zero_division=0),
        'precision_micro': precision_score(labels, pred_binary, average='micro', zero_division=0),
        'precision_macro': precision_score(labels, pred_binary, average='macro', zero_division=0),
        'recall_micro': recall_score(labels, pred_binary, average='micro', zero_division=0),
        'recall_macro': recall_score(labels, pred_binary, average='macro', zero_division=0),
    }

    # Calculate per-class metrics
    f1_per_class = f1_score(labels, pred_binary, average=None, zero_division=0)

    # Distribution metrics
    metrics['avg_predictions_per_sample'] = pred_binary.sum(axis=1).mean()
    metrics['avg_labels_per_sample'] = labels.sum(axis=1).mean()
    metrics['num_samples'] = len(labels)
    metrics['num_classes'] = labels.shape[1]

    # Class-wise statistics
    active_classes = (labels.sum(axis=0) > 0).sum()
    predicted_classes = (pred_binary.sum(axis=0) > 0).sum()

    metrics['active_classes'] = int(active_classes)
    metrics['predicted_classes'] = int(predicted_classes)
    metrics['avg_f1_per_class'] = f1_per_class.mean()
    metrics['std_f1_per_class'] = f1_per_class.std()

    return metrics, f1_per_class


def optimize_thresholds(predictions, labels, thresholds=None):
    """
    Find optimal thresholds per class to maximize F1

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        thresholds: List of thresholds to try (default: 0.1 to 0.9)

    Returns:
        optimal_thresholds: Array of optimal thresholds per class
        best_f1: Best F1 score achieved
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    n_classes = predictions.shape[1]
    optimal_thresholds = np.ones(n_classes) * 0.5  # Default to 0.5

    print("\nOptimizing thresholds per class...")

    for class_idx in tqdm(range(n_classes), desc="Optimizing"):
        best_f1 = 0
        best_thresh = 0.5

        # Only optimize if there are positive samples
        if labels[:, class_idx].sum() == 0:
            continue

        for thresh in thresholds:
            preds_binary = (predictions[:, class_idx] > thresh).astype(int)
            f1 = f1_score(labels[:, class_idx], preds_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds[class_idx] = best_thresh

    # Calculate F1 with optimized thresholds
    pred_binary = np.zeros_like(predictions)
    for i in range(n_classes):
        pred_binary[:, i] = (predictions[:, i] > optimal_thresholds[i]).astype(int)

    best_f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
    best_f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)

    print(f"Optimized F1 Micro: {best_f1_micro:.4f}")
    print(f"Optimized F1 Macro: {best_f1_macro:.4f}")

    return optimal_thresholds, best_f1_micro


def main():
    parser = argparse.ArgumentParser(description='Evaluate CAFA-6 model')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    parser.add_argument('--optimize_threshold', action='store_true',
                       help='Optimize thresholds per class')
    parser.add_argument('--multitask', action='store_true',
                       help='Multi-task model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("CAFA-6 Model Evaluation")
    print(f"{'='*60}\n")

    # Load embeddings and labels
    print("Loading data...")
    with open(args.embeddings, 'rb') as f:
        emb_data = pickle.load(f)
    with open(args.labels, 'rb') as f:
        label_data = pickle.load(f)

    # Create dataset
    if args.multitask:
        aspect_labels = {
            'C': label_data['aspect_labels']['C'][0],
            'F': label_data['aspect_labels']['F'][0],
            'P': label_data['aspect_labels']['P'][0]
        }
        dataset = MultiAspectEmbeddingDataset(
            protein_ids=label_data['protein_ids'],
            embeddings=emb_data['embeddings'],
            aspect_labels=aspect_labels
        )
        num_classes_C = aspect_labels['C'].shape[1]
        num_classes_F = aspect_labels['F'].shape[1]
        num_classes_P = aspect_labels['P'].shape[1]
    else:
        dataset = ProteinEmbeddingDataset(
            protein_ids=label_data['protein_ids'],
            embeddings=emb_data['embeddings'],
            labels=label_data['full_labels']
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=args.device)

    if args.multitask:
        model = MultiTaskProteinClassifier(
            embedding_dim=emb_data['embedding_dim'],
            num_classes_C=num_classes_C,
            num_classes_F=num_classes_F,
            num_classes_P=num_classes_P
        )
    else:
        model = SimpleProteinClassifier(
            embedding_dim=emb_data['embedding_dim'],
            num_classes=label_data['full_labels'].shape[1]
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    print(f"Model loaded from: {args.model}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, dataloader, args.device, args.multitask)

    # Calculate metrics
    print("\nCalculating metrics...")

    if args.multitask:
        metrics_C, f1_per_class_C = calculate_metrics(
            results['predictions_C'], results['labels_C'], args.threshold
        )
        metrics_F, f1_per_class_F = calculate_metrics(
            results['predictions_F'], results['labels_F'], args.threshold
        )
        metrics_P, f1_per_class_P = calculate_metrics(
            results['predictions_P'], results['labels_P'], args.threshold
        )

        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"\nCellular Component (C):")
        print(f"  F1 Micro: {metrics_C['f1_micro']:.4f}")
        print(f"  F1 Macro: {metrics_C['f1_macro']:.4f}")
        print(f"  Precision: {metrics_C['precision_micro']:.4f}")
        print(f"  Recall: {metrics_C['recall_micro']:.4f}")

        print(f"\nMolecular Function (F):")
        print(f"  F1 Micro: {metrics_F['f1_micro']:.4f}")
        print(f"  F1 Macro: {metrics_F['f1_macro']:.4f}")
        print(f"  Precision: {metrics_F['precision_micro']:.4f}")
        print(f"  Recall: {metrics_F['recall_micro']:.4f}")

        print(f"\nBiological Process (P):")
        print(f"  F1 Micro: {metrics_P['f1_micro']:.4f}")
        print(f"  F1 Macro: {metrics_P['f1_macro']:.4f}")
        print(f"  Precision: {metrics_P['precision_micro']:.4f}")
        print(f"  Recall: {metrics_P['recall_micro']:.4f}")

        avg_f1 = (metrics_C['f1_micro'] + metrics_F['f1_micro'] + metrics_P['f1_micro']) / 3
        print(f"\nAverage F1 Micro: {avg_f1:.4f}")

        # Save metrics
        all_metrics = {
            'C': metrics_C,
            'F': metrics_F,
            'P': metrics_P,
            'avg_f1_micro': avg_f1
        }
    else:
        metrics, f1_per_class = calculate_metrics(
            results['predictions'], results['labels'], args.threshold
        )

        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"\nF1 Micro: {metrics['f1_micro']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"Precision Micro: {metrics['precision_micro']:.4f}")
        print(f"Recall Micro: {metrics['recall_micro']:.4f}")
        print(f"\nAvg predictions/sample: {metrics['avg_predictions_per_sample']:.2f}")
        print(f"Avg labels/sample: {metrics['avg_labels_per_sample']:.2f}")
        print(f"Active classes: {metrics['active_classes']}")
        print(f"Predicted classes: {metrics['predicted_classes']}")

        all_metrics = metrics

        # Optimize thresholds if requested
        if args.optimize_threshold:
            optimal_thresholds, best_f1 = optimize_thresholds(
                results['predictions'], results['labels']
            )
            np.save(output_dir / 'optimal_thresholds.npy', optimal_thresholds)
            print(f"\nOptimal thresholds saved to: {output_dir / 'optimal_thresholds.npy'}")

    # Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in all_metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
                                    for k, v in value.items()}
            else:
                json_metrics[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        json.dump(json_metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_dir / 'metrics.json'}")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
