"""Inspect the cloned ASL Citizen processed dataset without creating splits.

The script supports the repository's sharded NumPy layout and also provides
basic inspection for consolidated NumPy, Parquet, and Hugging Face datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
SIGNER_KEY_TERMS = ("signer", "subject", "participant", "creator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ASL dataset splits without making random splits."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset directory (default: directory containing this script).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def shape_text(shape: tuple[int, ...] | None) -> str:
    return str(tuple(shape)) if shape is not None else "unavailable"


def inspect_numpy_split(root: Path, split: str) -> tuple[tuple[int, ...] | None, np.ndarray]:
    """Return the feature shape and all labels for a split.

    Labels are small, so collecting them is inexpensive. Feature arrays are
    opened with mmap to avoid loading the complete dataset into memory.
    """
    shard_dir = root / "shards"
    feature_shards = sorted(shard_dir.glob(f"{split}_features_*.npy"))
    label_shards = sorted(shard_dir.glob(f"{split}_labels_*.npy"))

    if feature_shards or label_shards:
        if not feature_shards or len(feature_shards) != len(label_shards):
            raise ValueError(
                f"{split}: feature/label shard counts do not match "
                f"({len(feature_shards)} vs {len(label_shards)})."
            )
        total_rows = 0
        feature_shape: tuple[int, ...] | None = None
        labels: list[np.ndarray] = []
        for feature_path, label_path in zip(feature_shards, label_shards):
            features = np.load(feature_path, mmap_mode="r")
            shard_labels = np.asarray(np.load(label_path))
            if features.ndim != 3:
                raise ValueError(f"{feature_path} has shape {features.shape}; expected 3-D.")
            if features.shape[0] != shard_labels.shape[0]:
                raise ValueError(f"{feature_path} and {label_path} have different row counts.")
            if feature_shape is None:
                feature_shape = (0, *features.shape[1:])
            elif feature_shape[1:] != features.shape[1:]:
                raise ValueError(f"{split} feature shards have inconsistent dimensions.")
            total_rows += features.shape[0]
            labels.append(shard_labels.reshape(-1))
        return (total_rows, *feature_shape[1:]), np.concatenate(labels)

    feature_path = root / f"{split}_features.npy"
    label_path = root / f"{split}_labels.npy"
    if feature_path.exists() and label_path.exists():
        features = np.load(feature_path, mmap_mode="r")
        labels = np.asarray(np.load(label_path)).reshape(-1)
        if features.ndim != 3:
            raise ValueError(f"{feature_path} has shape {features.shape}; expected 3-D.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"{feature_path} and {label_path} have different row counts.")
        return tuple(features.shape), labels

    raise FileNotFoundError(f"No NumPy files found for the {split!r} split.")


def inspect_tabular_split(root: Path, split: str) -> tuple[tuple[int, ...] | None, np.ndarray]:
    """Inspect a Parquet or Hugging Face split when no NumPy files are present."""
    parquet_paths = sorted(root.glob(f"{split}*.parquet"))
    if parquet_paths:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("Install pyarrow to inspect Parquet datasets.") from error
        table = parquet.read_table(parquet_paths[0])
        columns = table.column_names
        feature_column = next(
            (name for name in columns if name in {"features", "input_features", "sequence"}),
            None,
        )
        label_column = next((name for name in columns if name in {"label", "labels"}), None)
        if feature_column is None or label_column is None:
            raise ValueError(f"{parquet_paths[0]} must contain feature and label columns.")
        features = table[feature_column].to_pylist()
        labels = np.asarray(table[label_column].to_pylist()).reshape(-1)
        feature_shape = (len(features), *np.asarray(features[0]).shape)
        return feature_shape, labels

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "Install datasets to inspect a Hugging Face dataset saved with load_from_disk."
        ) from error
    dataset = load_from_disk(str(root))
    split_dataset = dataset[split] if hasattr(dataset, "keys") else dataset
    labels = np.asarray(split_dataset["label"])
    feature_values = split_dataset["features"]
    feature_shape = (len(feature_values), *np.asarray(feature_values[0]).shape)
    return feature_shape, labels


def inspect_split(root: Path, split: str) -> tuple[tuple[int, ...] | None, np.ndarray]:
    try:
        return inspect_numpy_split(root, split)
    except FileNotFoundError:
        return inspect_tabular_split(root, split)


def iter_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from iter_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_keys(child)


def signer_evidence(root: Path) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    manifests = sorted((root / "metadata").glob("*_manifest.csv"))
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as file:
            fieldnames = next(csv.reader(file), [])
        signer_fields = [
            field for field in fieldnames if any(term in field.lower() for term in SIGNER_KEY_TERMS)
        ]
        if signer_fields:
            evidence.append(f"{manifest.name}: {', '.join(signer_fields)}")

    for metadata_path in sorted(root.glob("*.json")) + sorted((root / "metadata").glob("*.json")):
        metadata = load_json(metadata_path)
        signer_keys = [
            text
            for text in iter_keys(metadata)
            if any(term in text.lower() for term in SIGNER_KEY_TERMS)
        ]
        if signer_keys:
            evidence.append(f"{metadata_path.name}: {', '.join(sorted(set(signer_keys)))}")
    return bool(evidence), evidence


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    print(f"Dataset root: {root}")
    print("Random splits created: NO (existing train/val/test files are inspected only)")

    split_results: dict[str, tuple[tuple[int, ...] | None, np.ndarray]] = {}
    for split in SPLITS:
        shape, labels = inspect_split(root, split)
        split_results[split] = shape, labels
        print(f"{split.capitalize():>10} features: {shape_text(shape)}")
        print(f"{split.capitalize():>10} labels:   {shape_text(labels.shape)}")

    class_ids = np.unique(np.concatenate([labels for _, labels in split_results.values()]))
    print(f"Unique class IDs: {len(class_ids)}")
    print(f"Expected 200 classes: {'PASS' if len(class_ids) == 200 else 'FAIL'}")

    signer_metadata_found, evidence = signer_evidence(root)
    if signer_metadata_found:
        print("Signer-independent split metadata: FOUND")
        for item in evidence:
            print(f"  Evidence: {item}")
    else:
        print(
            "Signer-independent split metadata: NOT CONFIRMED "
            "(no signer/subject/participant field was found)"
        )
        print("The script will not infer signer independence from folder names alone.")


if __name__ == "__main__":
    main()
