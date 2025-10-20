"""
Training script for CAFA-6 baseline model
Uses pre-computed ESM-2 embeddings with simple dense neural network

Usage:
    python scripts/train_baseline.py \
        --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
        --labels data/processed/preprocessed_data.pkl \
        --output_dir models/baseline \
        --epochs 10 \
        --batch_size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import SimpleProteinClassifier
from src.data import ProteinEmbeddingDataset
from src.utils import FocalLoss, WeightedBCELoss, get_loss_function


class BaselineTrainer:
    """
    Trainer for baseline protein function prediction model
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 output_dir: str = 'models/baseline'):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: 'cuda' or 'cpu'
            output_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_micro': [],
            'val_f1_macro': []
        }

        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            embeddings = batch['embedding'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> dict:
        """
        Validate model

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                embeddings = batch['embedding'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)

                # Get predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_predictions.append(probs)
                all_labels.append(labels_np)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Concatenate all batches
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)

        # Calculate metrics
        avg_loss = total_loss / num_batches
        metrics = self._calculate_metrics(predictions, labels)
        metrics['val_loss'] = avg_loss

        return metrics

    def _calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray,
                          threshold: float = 0.5) -> dict:
        """
        Calculate F1 scores

        Args:
            predictions: Predicted probabilities (n_samples, n_classes)
            labels: Ground truth labels (n_samples, n_classes)
            threshold: Threshold for binary predictions

        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Binarize predictions
        pred_binary = (predictions > threshold).astype(int)

        # Calculate metrics
        f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
        f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        precision = precision_score(labels, pred_binary, average='micro', zero_division=0)
        recall = recall_score(labels, pred_binary, average='micro', zero_division=0)

        # Calculate per-sample metrics
        n_samples = len(labels)
        correct_labels = (pred_binary == labels).sum()
        total_labels = labels.size

        metrics = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
            'accuracy': correct_labels / total_labels,
            'avg_predictions_per_sample': pred_binary.sum(axis=1).mean(),
            'avg_labels_per_sample': labels.sum(axis=1).mean()
        }

        return metrics

    def train(self, num_epochs: int, scheduler=None, save_best: bool = True):
        """
        Train for multiple epochs

        Args:
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler (optional)
            save_best: Whether to save best model
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_f1_micro'].append(val_metrics['f1_micro'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])

            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val F1 Micro: {val_metrics['f1_micro']:.4f}")
            print(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Avg Predictions/Sample: {val_metrics['avg_predictions_per_sample']:.2f}")
            print(f"  Avg Labels/Sample: {val_metrics['avg_labels_per_sample']:.2f}")

            # Save best model
            if save_best and val_metrics['f1_micro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_micro']
                self.best_epoch = epoch + 1

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1_micro': val_metrics['f1_micro'],
                    'val_f1_macro': val_metrics['f1_macro'],
                    'val_loss': val_metrics['val_loss']
                }

                checkpoint_path = self.output_dir / 'best_model.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"\n  ✓ New best model saved! (F1: {val_metrics['f1_micro']:.4f})")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(final_checkpoint, self.output_dir / 'final_model.pt')

        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best F1 Micro: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
        print(f"Models saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train baseline CAFA-6 model')

    # Data
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings pickle file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to preprocessed labels pickle file')

    # Model
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')

    # Loss
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['focal', 'bce', 'weighted_bce'],
                       help='Loss function')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['none', 'plateau', 'cosine', 'step'],
                       help='Learning rate scheduler')

    # Other
    parser.add_argument('--output_dir', type=str, default='models/baseline',
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"\n{'='*60}")
    print("CAFA-6 Baseline Training")
    print(f"{'='*60}")

    # Load embeddings
    print("\nLoading embeddings...")
    with open(args.embeddings, 'rb') as f:
        emb_data = pickle.load(f)

    embeddings = emb_data['embeddings']
    emb_protein_ids = emb_data['protein_ids']
    embedding_dim = emb_data['embedding_dim']

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Embedding dim: {embedding_dim}")

    # Load labels
    print("\nLoading labels...")
    with open(args.labels, 'rb') as f:
        label_data = pickle.load(f)

    labels = label_data['full_labels']
    label_protein_ids = label_data['protein_ids']
    num_classes = labels.shape[1]

    print(f"  Labels: {labels.shape}")
    print(f"  Num classes: {num_classes}")

    # Align protein IDs (ensure same order)
    print("\nAligning protein IDs...")
    common_ids = set(emb_protein_ids) & set(label_protein_ids)
    print(f"  Common proteins: {len(common_ids)}")

    # Get indices
    emb_indices = [emb_protein_ids.index(pid) for pid in common_ids]
    label_indices = [label_protein_ids.index(pid) for pid in common_ids]

    embeddings = embeddings[emb_indices]
    labels = labels[label_indices]
    protein_ids = list(common_ids)

    print(f"  Final dataset size: {len(protein_ids)}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = ProteinEmbeddingDataset(
        protein_ids=protein_ids,
        embeddings=embeddings,
        labels=labels
    )

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"  Train size: {train_size}")
    print(f"  Val size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = SimpleProteinClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SimpleProteinClassifier")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Total parameters: {num_params:,}")

    # Create loss function
    print(f"\nCreating loss function: {args.loss}")
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"  Alpha: {args.focal_alpha}, Gamma: {args.focal_gamma}")
    elif args.loss == 'weighted_bce':
        class_weights = torch.FloatTensor(label_data['class_weights'])
        criterion = WeightedBCELoss(class_weights=class_weights)
        print(f"  Using class weights")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Create optimizer
    print(f"\nCreating optimizer: {args.optimizer}")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    # Create scheduler
    scheduler = None
    if args.scheduler != 'none':
        print(f"\nCreating scheduler: {args.scheduler}")
        if args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )

    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        output_dir=args.output_dir
    )

    # Train
    trainer.train(num_epochs=args.epochs, scheduler=scheduler)

    # Save config
    config = vars(args)
    config['embedding_dim'] = embedding_dim
    config['num_classes'] = num_classes
    config['train_size'] = train_size
    config['val_size'] = val_size

    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Configuration saved to config.json")


if __name__ == '__main__':
    main()
