"""Train a BiGRU with attention on the provided ASL Citizen splits."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


INPUT_DIM = 450
NUM_CLASSES = 200
HIDDEN_SIZE = 256
BATCH_SIZE = 32
DROPOUT = 0.35
LEARNING_RATE = 3e-4


@dataclass(frozen=True)
class ShardIndex:
    feature_path: Path
    label_path: Path
    start: int
    length: int


class ShardedNumpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Index sharded arrays without concatenating the complete dataset in RAM."""

    def __init__(self, root: Path, split: str) -> None:
        self.shards = self._find_shards(root, split)
        self._features: dict[Path, np.ndarray] = {}
        self._labels: dict[Path, np.ndarray] = {}
        self._length = sum(shard.length for shard in self.shards)
        if self._length == 0:
            raise ValueError(f"{split} split is empty.")

    @staticmethod
    def _find_shards(root: Path, split: str) -> list[ShardIndex]:
        shard_dir = root / "shards"
        feature_paths = sorted(shard_dir.glob(f"{split}_features_*.npy"))
        label_paths = sorted(shard_dir.glob(f"{split}_labels_*.npy"))
        if not feature_paths or len(feature_paths) != len(label_paths):
            raise FileNotFoundError(
                f"Expected matching {split} feature and label shards in {shard_dir}."
            )

        shards: list[ShardIndex] = []
        start = 0
        for feature_path, label_path in zip(feature_paths, label_paths):
            features = np.load(feature_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            if features.ndim != 3 or features.shape[1:] != (200, INPUT_DIM):
                raise ValueError(
                    f"{feature_path} has shape {features.shape}; expected "
                    f"(N, 200, {INPUT_DIM})."
                )
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError(f"{feature_path} and {label_path} have incompatible shapes.")
            shards.append(ShardIndex(feature_path, label_path, start, features.shape[0]))
            start += features.shape[0]
        return shards

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[ShardIndex, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for shard in self.shards:
            if shard.start <= index < shard.start + shard.length:
                return shard, index - shard.start
        raise IndexError(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard, row = self._locate(index)
        if shard.feature_path not in self._features:
            self._features[shard.feature_path] = np.load(shard.feature_path, mmap_mode="r")
            self._labels[shard.label_path] = np.load(shard.label_path, mmap_mode="r")
        features = np.asarray(self._features[shard.feature_path][row], dtype=np.float32)
        label = int(self._labels[shard.label_path][row])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


class AttentionPooling(nn.Module):
    def __init__(self, input_size: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_size, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class BiGRUAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = AttentionPooling(hidden_size * 2, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(self.input_norm(inputs))
        sequence, _ = self.gru(projected)
        return self.classifier(self.attention(sequence))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/best_bigru_attention_model.pt"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if is_training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


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

    model = BiGRUAttentionClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 2,
                "bidirectional": True,
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
