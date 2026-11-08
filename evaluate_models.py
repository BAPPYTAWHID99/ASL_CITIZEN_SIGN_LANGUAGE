"""Evaluate the trained ASL models once on the official held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from train_bigru import BiGRUAttentionClassifier, ShardedNumpyDataset
from train_transformer import TransformerClassifier


NUM_CLASSES = 200
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--bigru-checkpoint",
        type=Path,
        default=Path("models/best_bigru_attention_model.pt"),
    )
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=Path("models/best_transformer_model.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metrics-output", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--plot-output", type=Path, default=Path("confusion_matrix.png"))
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {path}")
    return checkpoint


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    top_five_predictions: np.ndarray,
) -> dict[str, object]:
    top_one = float(np.mean(predictions == labels))
    top_five = float(np.mean(np.any(top_five_predictions == labels[:, None], axis=1)))
    macro_f1 = float(
        f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)
    )
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "sample_count": int(labels.size),
        "top_1_accuracy": top_one,
        "top_5_accuracy": top_five,
        "macro_f1_score": macro_f1,
        "per_class_f1_score": {
            str(class_id): float(report[str(class_id)]["f1-score"])
            for class_id in range(NUM_CLASSES)
        },
    }


def save_confusion_plot(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    matrices = {
        name: confusion_matrix(labels, values, labels=np.arange(NUM_CLASSES))
        for name, values in predictions.items()
    }
    figure, axes = plt.subplots(1, len(matrices), figsize=(22, 10), squeeze=False)
    for axis, (name, matrix) in zip(axes[0], matrices.items()):
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(f"{name} confusion matrix")
        axis.set_xlabel("Predicted class ID")
        axis.set_ylabel("True class ID")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    root = args.dataset_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bigru_checkpoint = load_checkpoint(resolve_path(root, args.bigru_checkpoint), device)
    transformer_checkpoint = load_checkpoint(
        resolve_path(root, args.transformer_checkpoint), device
    )
    bigru = BiGRUAttentionClassifier().to(device)
    transformer = TransformerClassifier().to(device)
    bigru.load_state_dict(bigru_checkpoint["model_state_dict"])
    transformer.load_state_dict(transformer_checkpoint["model_state_dict"])
    bigru.eval()
    transformer.eval()

    print(f"Device: {device}")
    split_results: dict[str, dict[str, object]] = {}
    test_labels: np.ndarray | None = None
    test_predictions: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        dataset = ShardedNumpyDataset(root, split)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        labels: list[np.ndarray] = []
        predictions: dict[str, list[np.ndarray]] = {
            "bigru_attention": [],
            "transformer": [],
        }
        top_five: dict[str, list[np.ndarray]] = {
            "bigru_attention": [],
            "transformer": [],
        }
        print(f"Evaluating {split} split ({len(dataset)} samples).")
        with torch.inference_mode():
            for features, batch_labels in loader:
                features = features.to(device)
                bigru_logits = bigru(features)
                transformer_logits = transformer(features)
                labels.append(batch_labels.numpy())
                for name, logits in (
                    ("bigru_attention", bigru_logits),
                    ("transformer", transformer_logits),
                ):
                    predictions[name].append(logits.argmax(dim=1).cpu().numpy())
                    top_five[name].append(logits.topk(5, dim=1).indices.cpu().numpy())

        all_labels = np.concatenate(labels)
        prediction_arrays = {
            name: np.concatenate(values) for name, values in predictions.items()
        }
        top_five_arrays = {
            name: np.concatenate(values) for name, values in top_five.items()
        }
        split_results[split] = {
            name: calculate_metrics(all_labels, prediction_arrays[name], top_five_arrays[name])
            for name in prediction_arrays
        }
        if split == "test":
            test_labels = all_labels
            test_predictions = prediction_arrays

    if test_labels is None:
        raise RuntimeError("Test split evaluation did not produce labels.")
    validation_results = split_results["val"]
    test_results = split_results["test"]
    metrics = {
        "validation": validation_results,
        "test": test_results,
        "accuracy_gap_test_minus_validation": {
            name: float(
                test_results[name]["top_1_accuracy"]  # type: ignore[index]
                - validation_results[name]["top_1_accuracy"]  # type: ignore[index]
            )
            for name in test_results
        },
    }

    metrics_path = resolve_path(root, args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "num_classes": NUM_CLASSES,
                "evaluated_once": True,
                "splits": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_path = resolve_path(root, args.plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_confusion_plot(test_labels, test_predictions, plot_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {plot_path}")


if __name__ == "__main__":
    main()
