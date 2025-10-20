"""
Baseline model architectures for CAFA-6 protein function prediction
"""

import torch
import torch.nn as nn
from typing import Optional


class SimpleProteinClassifier(nn.Module):
    """
    Simple baseline classifier using pre-computed embeddings
    Good for quick experimentation and establishing baseline
    """

    def __init__(self,
                 embedding_dim: int,
                 num_classes: int,
                 hidden_dims: list = [1024, 512],
                 dropout: float = 0.3):
        """
        Args:
            embedding_dim: Dimension of input protein embeddings
            num_classes: Number of GO terms to predict
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        return self.classifier(embeddings)


class ESM2Classifier(nn.Module):
    """
    Classifier using ESM-2 protein language model
    Can use frozen or fine-tuned ESM-2 encoder
    """

    def __init__(self,
                 esm_model_name: str = 'esm2_t33_650M_UR50D',
                 num_classes: int = 26125,
                 hidden_dim: int = 1024,
                 dropout: float = 0.3,
                 freeze_encoder: bool = True,
                 num_unfrozen_layers: int = 0):
        """
        Args:
            esm_model_name: ESM model name from torch.hub
            num_classes: Number of GO terms to predict
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            freeze_encoder: If True, freeze ESM encoder weights
            num_unfrozen_layers: Number of top encoder layers to fine-tune (if not fully frozen)
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
            # Freeze all layers first
            for param in self.esm_model.parameters():
                param.requires_grad = False
            # Unfreeze last N layers
            total_layers = len(self.esm_model.layers)
            for i in range(total_layers - num_unfrozen_layers, total_layers):
                for param in self.esm_model.layers[i].parameters():
                    param.requires_grad = True

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length) - optional

        Returns:
            logits: (batch_size, num_classes)
        """
        # Get ESM embeddings
        with torch.set_grad_enabled(not self.esm_model.training or
                                    any(p.requires_grad for p in self.esm_model.parameters())):
            results = self.esm_model(input_ids, repr_layers=[33])  # Get last layer
            embeddings = results['representations'][33]

        # Use [CLS] token embedding (position 0)
        cls_embedding = embeddings[:, 0, :]

        # Classify
        logits = self.classifier(cls_embedding)

        return logits


class ProtBERTClassifier(nn.Module):
    """
    Classifier using ProtBERT protein language model from HuggingFace
    """

    def __init__(self,
                 model_name: str = 'Rostlab/prot_bert',
                 num_classes: int = 26125,
                 hidden_dim: int = 1024,
                 dropout: float = 0.3,
                 freeze_encoder: bool = True):
        """
        Args:
            model_name: HuggingFace model name
            num_classes: Number of GO terms to predict
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            freeze_encoder: If True, freeze BERT encoder weights
        """
        super().__init__()

        try:
            from transformers import BertModel, BertTokenizer
            self.encoder = BertModel.from_pretrained(model_name)
            embedding_dim = self.encoder.config.hidden_size
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Get BERT embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classify
        logits = self.classifier(cls_embedding)

        return logits


class CNNProteinClassifier(nn.Module):
    """
    CNN-based classifier for protein sequences
    Good for capturing local motifs and patterns
    """

    def __init__(self,
                 vocab_size: int = 26,  # 20 amino acids + special tokens
                 embedding_dim: int = 128,
                 num_classes: int = 26125,
                 num_filters: list = [256, 256, 256],
                 kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.3):
        """
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Dimension of amino acid embeddings
            num_classes: Number of GO terms to predict
            num_filters: Number of filters for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Multiple 1D convolutions with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters[i], kernel_sizes[i], padding=kernel_sizes[i]//2)
            for i in range(len(kernel_sizes))
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters[i]) for i in range(len(kernel_sizes))
        ])

        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        total_filters = sum(num_filters)
        self.fc = nn.Sequential(
            nn.Linear(total_filters, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed sequences
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Transpose for conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply convolutions
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(embedded)
            x = bn(x)
            x = torch.relu(x)
            x = self.pool(x).squeeze(-1)  # (batch, num_filters)
            conv_outputs.append(x)

        # Concatenate all conv outputs
        concat = torch.cat(conv_outputs, dim=1)  # (batch, total_filters)

        # Classify
        logits = self.fc(concat)

        return logits


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get model by name

    Args:
        model_type: One of 'simple', 'esm2', 'protbert', 'cnn'
        **kwargs: Model-specific arguments

    Returns:
        PyTorch model
    """
    models = {
        'simple': SimpleProteinClassifier,
        'esm2': ESM2Classifier,
        'protbert': ProtBERTClassifier,
        'cnn': CNNProteinClassifier
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)


if __name__ == '__main__':
    # Test models
    batch_size = 4
    seq_length = 512
    embedding_dim = 1280
    num_classes = 26125

    # Test simple classifier
    print("Testing SimpleProteinClassifier...")
    model = SimpleProteinClassifier(embedding_dim, num_classes)
    embeddings = torch.randn(batch_size, embedding_dim)
    output = model(embeddings)
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test CNN classifier
    print("\nTesting CNNProteinClassifier...")
    model = CNNProteinClassifier(num_classes=num_classes)
    input_ids = torch.randint(0, 26, (batch_size, seq_length))
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
