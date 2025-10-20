"""
Training script for CAFA-6 multi-task model
Separate prediction heads for C, F, P aspects

Usage:
    python scripts/train_multitask.py \
        --embeddings data/embeddings/train_embeddings_esm2_t30_150M_UR50D_balanced.pkl \
        --labels data/processed/preprocessed_data.pkl \
        --output_dir models/multitask \
        --epochs 15
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

from src.models import MultiTaskProteinClassifier
from src.data import MultiAspectEmbeddingDataset
from src.utils import MultiTaskLoss


class MultiTaskTrainer:
    """
    Trainer for multi-task protein function prediction model
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 output_dir: str = 'models/multitask'):
        """
        Args:
            model: Multi-task PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Multi-task loss function
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
            'train_loss_C': [],
            'train_loss_F': [],
            'train_loss_P': [],
            'val_loss': [],
            'val_loss_C': [],
            'val_loss_F': [],
            'val_loss_P': [],
            'val_f1_C': [],
            'val_f1_F': [],
            'val_f1_P': [],
            'val_f1_avg': []
        }

        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_loss_C = 0.0
        total_loss_F = 0.0
        total_loss_P = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            embeddings = batch['embedding'].to(self.device)
            labels_C = batch['labels_C'].to(self.device)
            labels_F = batch['labels_F'].to(self.device)
            labels_P = batch['labels_P'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)

            # Calculate loss
            targets = {'C': labels_C, 'F': labels_F, 'P': labels_P}
            loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss_dict['total_loss']
            total_loss_C += loss_dict['loss_C']
            total_loss_F += loss_dict['loss_F']
            total_loss_P += loss_dict['loss_P']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'C': f"{loss_dict['loss_C']:.4f}",
                'F': f"{loss_dict['loss_F']:.4f}",
                'P': f"{loss_dict['loss_P']:.4f}"
            })

        return {
            'total_loss': total_loss / num_batches,
            'loss_C': total_loss_C / num_batches,
            'loss_F': total_loss_F / num_batches,
            'loss_P': total_loss_P / num_batches
        }

    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_loss_C = 0.0
        total_loss_F = 0.0
        total_loss_P = 0.0
        num_batches = 0

        all_preds_C, all_preds_F, all_preds_P = [], [], []
        all_labels_C, all_labels_F, all_labels_P = [], [], []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                embeddings = batch['embedding'].to(self.device)
                labels_C = batch['labels_C'].to(self.device)
                labels_F = batch['labels_F'].to(self.device)
                labels_P = batch['labels_P'].to(self.device)

                # Forward pass
                outputs = self.model(embeddings)

                # Calculate loss
                targets = {'C': labels_C, 'F': labels_F, 'P': labels_P}
                loss, loss_dict = self.criterion(outputs, targets)

                # Get predictions
                preds_C = torch.sigmoid(outputs['C']).cpu().numpy()
                preds_F = torch.sigmoid(outputs['F']).cpu().numpy()
                preds_P = torch.sigmoid(outputs['P']).cpu().numpy()

                all_preds_C.append(preds_C)
                all_preds_F.append(preds_F)
                all_preds_P.append(preds_P)

                all_labels_C.append(labels_C.cpu().numpy())
                all_labels_F.append(labels_F.cpu().numpy())
                all_labels_P.append(labels_P.cpu().numpy())

                total_loss += loss_dict['total_loss']
                total_loss_C += loss_dict['loss_C']
                total_loss_F += loss_dict['loss_F']
                total_loss_P += loss_dict['loss_P']
                num_batches += 1

        # Concatenate all batches
        preds_C = np.vstack(all_preds_C)
        preds_F = np.vstack(all_preds_F)
        preds_P = np.vstack(all_preds_P)

        labels_C = np.vstack(all_labels_C)
        labels_F = np.vstack(all_labels_F)
        labels_P = np.vstack(all_labels_P)

        # Calculate metrics for each aspect
        from sklearn.metrics import f1_score

        f1_C = f1_score(labels_C, (preds_C > 0.5).astype(int),
                       average='micro', zero_division=0)
        f1_F = f1_score(labels_F, (preds_F > 0.5).astype(int),
                       average='micro', zero_division=0)
        f1_P = f1_score(labels_P, (preds_P > 0.5).astype(int),
                       average='micro', zero_division=0)

        # Average F1 across aspects
        f1_avg = (f1_C + f1_F + f1_P) / 3

        return {
            'val_loss': total_loss / num_batches,
            'val_loss_C': total_loss_C / num_batches,
            'val_loss_F': total_loss_F / num_batches,
            'val_loss_P': total_loss_P / num_batches,
            'f1_C': f1_C,
            'f1_F': f1_F,
            'f1_P': f1_P,
            'f1_avg': f1_avg
        }

    def train(self, num_epochs: int, scheduler=None):
        """Train for multiple epochs"""
        print(f"\n{'='*60}")
        print(f"Starting multi-task training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()

            # Log metrics
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['train_loss_C'].append(train_metrics['loss_C'])
            self.history['train_loss_F'].append(train_metrics['loss_F'])
            self.history['train_loss_P'].append(train_metrics['loss_P'])

            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_loss_C'].append(val_metrics['val_loss_C'])
            self.history['val_loss_F'].append(val_metrics['val_loss_F'])
            self.history['val_loss_P'].append(val_metrics['val_loss_P'])

            self.history['val_f1_C'].append(val_metrics['f1_C'])
            self.history['val_f1_F'].append(val_metrics['f1_F'])
            self.history['val_f1_P'].append(val_metrics['f1_P'])
            self.history['val_f1_avg'].append(val_metrics['f1_avg'])

            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(C: {train_metrics['loss_C']:.4f}, "
                  f"F: {train_metrics['loss_F']:.4f}, "
                  f"P: {train_metrics['loss_P']:.4f})")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f} "
                  f"(C: {val_metrics['val_loss_C']:.4f}, "
                  f"F: {val_metrics['val_loss_F']:.4f}, "
                  f"P: {val_metrics['val_loss_P']:.4f})")
            print(f"  Val F1 - C: {val_metrics['f1_C']:.4f}, "
                  f"F: {val_metrics['f1_F']:.4f}, "
                  f"P: {val_metrics['f1_P']:.4f}")
            print(f"  Val F1 Avg: {val_metrics['f1_avg']:.4f}")

            # Save best model
            if val_metrics['f1_avg'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_avg']
                self.best_epoch = epoch + 1

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1_avg': val_metrics['f1_avg'],
                    'val_f1_C': val_metrics['f1_C'],
                    'val_f1_F': val_metrics['f1_F'],
                    'val_f1_P': val_metrics['f1_P']
                }

                torch.save(checkpoint, self.output_dir / 'best_model.pt')
                print(f"\n  ✓ New best model saved! (Avg F1: {val_metrics['f1_avg']:.4f})")

        # Save final model and history
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, self.output_dir / 'final_model.pt')

        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best Avg F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
        print(f"Models saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train multi-task CAFA-6 model')

    # Data
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)

    # Model
    parser.add_argument('--shared_hidden_dims', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('--aspect_hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)

    # Loss
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['focal', 'bce', 'asymmetric'])
    parser.add_argument('--learnable_weights', action='store_true',
                       help='Use learnable task weights')

    # Other
    parser.add_argument('--output_dir', type=str, default='models/multitask')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading embeddings and labels...")
    with open(args.embeddings, 'rb') as f:
        emb_data = pickle.load(f)
    with open(args.labels, 'rb') as f:
        label_data = pickle.load(f)

    # Get aspect-specific labels
    aspect_labels_data = label_data['aspect_labels']
    aspect_labels = {
        'C': aspect_labels_data['C'][0],
        'F': aspect_labels_data['F'][0],
        'P': aspect_labels_data['P'][0]
    }

    num_classes_C = aspect_labels['C'].shape[1]
    num_classes_F = aspect_labels['F'].shape[1]
    num_classes_P = aspect_labels['P'].shape[1]

    print(f"Num classes - C: {num_classes_C}, F: {num_classes_F}, P: {num_classes_P}")

    # Create dataset
    dataset = MultiAspectEmbeddingDataset(
        protein_ids=label_data['protein_ids'],
        embeddings=emb_data['embeddings'],
        aspect_labels=aspect_labels
    )

    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = MultiTaskProteinClassifier(
        embedding_dim=emb_data['embedding_dim'],
        num_classes_C=num_classes_C,
        num_classes_F=num_classes_F,
        num_classes_P=num_classes_P,
        shared_hidden_dims=args.shared_hidden_dims,
        aspect_hidden_dim=args.aspect_hidden_dim,
        dropout=args.dropout
    )

    # Loss
    criterion = MultiTaskLoss(
        num_classes_C=num_classes_C,
        num_classes_F=num_classes_F,
        num_classes_P=num_classes_P,
        loss_type=args.loss_type,
        learnable_weights=args.learnable_weights
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Trainer
    trainer = MultiTaskTrainer(
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


if __name__ == '__main__':
    main()
