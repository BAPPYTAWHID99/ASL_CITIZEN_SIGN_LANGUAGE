"""Train a compact Transformer encoder on the provided ASL Citizen splits."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_bigru import (
    INPUT_DIM,
    NUM_CLASSES,
    ShardedNumpyDataset,
    run_epoch,
)


EMBED_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
FEEDFORWARD_DIM = 768
SEQUENCE_LENGTH = 200
BATCH_SIZE = 32
DROPOUT = 0.3
LEARNING_RATE = 2e-4


class TransformerClassifier(nn.Module):
    """Transformer encoder with a learnable classification token."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embed_dim: int = EMBED_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        sequence_length: int = SEQUENCE_LENGTH,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length + 1, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected inputs with shape (batch, {SEQUENCE_LENGTH}, {INPUT_DIM}); "
                f"received {tuple(inputs.shape)}."
            )
        projected = self.input_projection(self.input_norm(inputs))
        cls_tokens = self.cls_token.expand(projected.shape[0], -1, -1)
        encoded = torch.cat((cls_tokens, projected), dim=1)
        encoded = encoded + self.position_embedding[:, : encoded.shape[1]]
        encoded = self.encoder(encoded)
        return self.classifier(encoded[:, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/best_transformer_model.pt"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    seed_everything(args.seed)
    root = args.dataset_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ShardedNumpyDataset(root, "train")
    val_dataset = ShardedNumpyDataset(root, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TransformerClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint.")
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "input_dim": INPUT_DIM,
                "embed_dim": EMBED_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "feedforward_dim": FEEDFORWARD_DIM,
                "sequence_length": SEQUENCE_LENGTH,
                "num_classes": NUM_CLASSES,
                "dropout": DROPOUT,
            },
            "best_val_loss": best_loss,
        },
        output_path,
    )
    print(f"Saved best weights to: {output_path}")


if __name__ == "__main__":
    main()
