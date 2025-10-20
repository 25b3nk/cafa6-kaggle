"""
Multi-task model architectures for CAFA-6 protein function prediction
Separate prediction heads for C, F, P aspects
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class MultiTaskProteinClassifier(nn.Module):
    """
    Multi-task classifier with separate heads for C, F, P aspects
    Uses pre-computed embeddings
    """

    def __init__(self,
                 embedding_dim: int,
                 num_classes_C: int,
                 num_classes_F: int,
                 num_classes_P: int,
                 shared_hidden_dims: list = [2048, 1024],
                 aspect_hidden_dim: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            embedding_dim: Dimension of input protein embeddings
            num_classes_C: Number of Cellular Component GO terms
            num_classes_F: Number of Molecular Function GO terms
            num_classes_P: Number of Biological Process GO terms
            shared_hidden_dims: Dimensions for shared encoder layers
            aspect_hidden_dim: Hidden dimension for aspect-specific heads
            dropout: Dropout probability
        """
        super().__init__()

        # Shared encoder
        shared_layers = []
        prev_dim = embedding_dim

        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim

        # Aspect-specific heads
        self.head_C = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_C, dropout
        )
        self.head_F = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_F, dropout
        )
        self.head_P = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_P, dropout
        )

    def _build_aspect_head(self, input_dim: int, hidden_dim: int,
                          num_classes: int, dropout: float) -> nn.Module:
        """Build a single aspect-specific prediction head"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            Dictionary with keys 'C', 'F', 'P' and logits as values
        """
        # Shared encoding
        shared_repr = self.shared_encoder(embeddings)

        # Aspect-specific predictions
        logits_C = self.head_C(shared_repr)
        logits_F = self.head_F(shared_repr)
        logits_P = self.head_P(shared_repr)

        return {
            'C': logits_C,
            'F': logits_F,
            'P': logits_P
        }


class MultiTaskESM2Classifier(nn.Module):
    """
    Multi-task classifier using ESM-2 encoder with separate heads for C, F, P
    """

    def __init__(self,
                 esm_model_name: str = 'esm2_t33_650M_UR50D',
                 num_classes_C: int = 2651,
                 num_classes_F: int = 6616,
                 num_classes_P: int = 16858,
                 shared_hidden_dim: int = 1024,
                 aspect_hidden_dim: int = 512,
                 dropout: float = 0.3,
                 freeze_encoder: bool = True,
                 num_unfrozen_layers: int = 0):
        """
        Args:
            esm_model_name: ESM model name
            num_classes_C/F/P: Number of GO terms per aspect
            shared_hidden_dim: Hidden dimension for shared layers
            aspect_hidden_dim: Hidden dimension for aspect-specific heads
            dropout: Dropout probability
            freeze_encoder: If True, freeze ESM encoder
            num_unfrozen_layers: Number of top layers to fine-tune
        """
        super().__init__()

        # Load ESM-2 model
        try:
            import esm
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
            embedding_dim = self.esm_model.embed_dim
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        elif num_unfrozen_layers > 0:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            total_layers = len(self.esm_model.layers)
            for i in range(total_layers - num_unfrozen_layers, total_layers):
                for param in self.esm_model.layers[i].parameters():
                    param.requires_grad = True

        # Shared projection layer
        self.shared_projection = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(shared_hidden_dim),
            nn.Dropout(dropout)
        )

        # Aspect-specific heads
        self.head_C = self._build_aspect_head(
            shared_hidden_dim, aspect_hidden_dim, num_classes_C, dropout
        )
        self.head_F = self._build_aspect_head(
            shared_hidden_dim, aspect_hidden_dim, num_classes_F, dropout
        )
        self.head_P = self._build_aspect_head(
            shared_hidden_dim, aspect_hidden_dim, num_classes_P, dropout
        )

    def _build_aspect_head(self, input_dim: int, hidden_dim: int,
                          num_classes: int, dropout: float) -> nn.Module:
        """Build a single aspect-specific prediction head"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length) - optional

        Returns:
            Dictionary with keys 'C', 'F', 'P' and logits as values
        """
        # Get ESM embeddings
        with torch.set_grad_enabled(not self.esm_model.training or
                                    any(p.requires_grad for p in self.esm_model.parameters())):
            results = self.esm_model(input_ids, repr_layers=[33])
            embeddings = results['representations'][33]

        # Use [CLS] token embedding (position 0)
        cls_embedding = embeddings[:, 0, :]

        # Shared projection
        shared_repr = self.shared_projection(cls_embedding)

        # Aspect-specific predictions
        logits_C = self.head_C(shared_repr)
        logits_F = self.head_F(shared_repr)
        logits_P = self.head_P(shared_repr)

        return {
            'C': logits_C,
            'F': logits_F,
            'P': logits_P
        }


class HierarchicalMultiTaskClassifier(nn.Module):
    """
    Multi-task classifier with hierarchical prediction
    First predicts aspect (C/F/P), then specific GO terms
    """

    def __init__(self,
                 embedding_dim: int,
                 num_classes_C: int,
                 num_classes_F: int,
                 num_classes_P: int,
                 shared_hidden_dims: list = [2048, 1024],
                 aspect_hidden_dim: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            embedding_dim: Dimension of input protein embeddings
            num_classes_C/F/P: Number of GO terms per aspect
            shared_hidden_dims: Dimensions for shared encoder
            aspect_hidden_dim: Hidden dimension for aspect heads
            dropout: Dropout probability
        """
        super().__init__()

        # Shared encoder
        shared_layers = []
        prev_dim = embedding_dim

        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim

        # Aspect presence prediction (auxiliary task)
        self.aspect_predictor = nn.Sequential(
            nn.Linear(shared_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # 3 aspects: C, F, P
        )

        # Aspect-specific GO term heads
        self.head_C = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_C, dropout
        )
        self.head_F = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_F, dropout
        )
        self.head_P = self._build_aspect_head(
            shared_output_dim, aspect_hidden_dim, num_classes_P, dropout
        )

    def _build_aspect_head(self, input_dim: int, hidden_dim: int,
                          num_classes: int, dropout: float) -> nn.Module:
        """Build a single aspect-specific prediction head"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            Dictionary with:
                - 'C', 'F', 'P': GO term logits
                - 'aspects': Aspect presence logits
        """
        # Shared encoding
        shared_repr = self.shared_encoder(embeddings)

        # Predict aspect presence
        aspect_logits = self.aspect_predictor(shared_repr)

        # Predict GO terms per aspect
        logits_C = self.head_C(shared_repr)
        logits_F = self.head_F(shared_repr)
        logits_P = self.head_P(shared_repr)

        return {
            'C': logits_C,
            'F': logits_F,
            'P': logits_P,
            'aspects': aspect_logits  # Auxiliary prediction
        }


def get_multitask_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get multi-task model by name

    Args:
        model_type: One of 'simple', 'esm2', 'hierarchical'
        **kwargs: Model-specific arguments

    Returns:
        PyTorch model
    """
    models = {
        'simple': MultiTaskProteinClassifier,
        'esm2': MultiTaskESM2Classifier,
        'hierarchical': HierarchicalMultiTaskClassifier
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)


if __name__ == '__main__':
    # Test multi-task models
    batch_size = 4
    embedding_dim = 1280
    num_classes_C = 2651
    num_classes_F = 6616
    num_classes_P = 16858

    # Test simple multi-task classifier
    print("Testing MultiTaskProteinClassifier...")
    model = MultiTaskProteinClassifier(
        embedding_dim,
        num_classes_C,
        num_classes_F,
        num_classes_P
    )
    embeddings = torch.randn(batch_size, embedding_dim)
    outputs = model(embeddings)

    print(f"Input shape: {embeddings.shape}")
    print(f"Output C shape: {outputs['C'].shape}")
    print(f"Output F shape: {outputs['F'].shape}")
    print(f"Output P shape: {outputs['P'].shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test hierarchical model
    print("\nTesting HierarchicalMultiTaskClassifier...")
    model = HierarchicalMultiTaskClassifier(
        embedding_dim,
        num_classes_C,
        num_classes_F,
        num_classes_P
    )
    outputs = model(embeddings)

    print(f"Input shape: {embeddings.shape}")
    print(f"Output C shape: {outputs['C'].shape}")
    print(f"Output F shape: {outputs['F'].shape}")
    print(f"Output P shape: {outputs['P'].shape}")
    print(f"Output aspects shape: {outputs['aspects'].shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
