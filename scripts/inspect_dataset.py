#!/usr/bin/env python3
"""
scripts/inspect_dataset.py

Inspects a dataset directory recursively to detect image formats,
annotation files, configuration YAMLs, split structure, and class names.

Usage:
    python scripts/inspect_dataset.py --dataset-dir dataset/raw
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_LABEL_EXTS = {".txt"}
CONFIG_EXTS = {".yaml", ".yml"}


def inspect_directory(base_dir: Path) -> dict:
    stats = {
        "exists": base_dir.exists(),
        "is_dir": base_dir.is_dir() if base_dir.exists() else False,
        "image_counts_by_ext": Counter(),
        "total_images": 0,
        "total_labels": 0,
        "config_files": [],
        "splits_found": {},
        "subdirs": [],
    }

    if not stats["exists"] or not stats["is_dir"]:
        return stats

    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            stats["subdirs"].append(item.name)

    # Search for known split directories
    for split in ["train", "valid", "val", "test"]:
        split_dir = base_dir / split
        if split_dir.exists() and split_dir.is_dir():
            split_images = 0
            split_labels = 0
            for file_path in split_dir.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in SUPPORTED_IMAGE_EXTS:
                        split_images += 1
                    elif ext in SUPPORTED_LABEL_EXTS:
                        split_labels += 1
            stats["splits_found"][split] = {
                "images": split_images,
                "labels": split_labels,
                "path": str(split_dir),
            }

    # Global scan
    for file_path in base_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                stats["image_counts_by_ext"][ext] += 1
                stats["total_images"] += 1
            elif ext in SUPPORTED_LABEL_EXTS:
                stats["total_labels"] += 1
            elif ext in CONFIG_EXTS:
                stats["config_files"].append(file_path)

    return stats


def parse_dataset_yaml(yaml_path: Path):
    if yaml is None:
        return {"error": "PyYAML is not installed. Run 'pip install PyYAML' to parse YAML files."}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect fish disease dataset directory structure and configurations."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/raw",
        help="Path to the dataset directory (default: dataset/raw)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    print("=" * 70)
    print(" AquaVigil - Dataset Inspector")
    print("=" * 70)
    print(f"Target Directory: {dataset_path.resolve()}\n")

    if not dataset_path.exists():
        print(f"[!] Directory does not exist: {dataset_path}")
        print("\nINSTRUCTIONS:")
        print("1. Download the fish disease dataset from Roboflow.")
        print(f"2. Extract its contents into: {dataset_path.resolve()}")
        print("3. Ensure that data.yaml and image folders are inside.")
        print(f"4. Re-run: python scripts/inspect_dataset.py --dataset-dir {args.dataset_dir}")
        sys.exit(1)

    stats = inspect_directory(dataset_path)

    print("--- Directory Structure ---")
    if stats["subdirs"]:
        print("Discovered Subdirectories:")
        for s in stats["subdirs"]:
            print(f"  - {s}/")
    else:
        print("  (No subdirectories found at root)")

    print("\n--- Split Analysis ---")
    if stats["splits_found"]:
        for split, info in stats["splits_found"].items():
            print(f"  [{split.upper()}]: {info['images']} images, {info['labels']} label files")
    else:
        print("  [!] No standard split directories (train/valid/test) detected at root level.")

    print("\n--- File Counts ---")
    print(f"  Total Images Found: {stats['total_images']}")
    for ext, count in stats["image_counts_by_ext"].items():
        print(f"    - {ext}: {count}")
    print(f"  Total Label Files (.txt): {stats['total_labels']}")

    print("\n--- Dataset Configurations ---")
    if stats["config_files"]:
        print(f"  Found {len(stats['config_files'])} configuration file(s):")
        for cfg in stats["config_files"]:
            print(f"\n  File: {cfg}")
            cfg_data = parse_dataset_yaml(cfg)
            if "error" in cfg_data:
                print(f"    [Error]: {cfg_data['error']}")
            else:
                names = cfg_data.get("names", None)
                nc = cfg_data.get("nc", None)
                print(f"    Number of classes (nc): {nc}")
                print(f"    Class names: {names}")
                print(f"    Train path: {cfg_data.get('train')}")
                print(f"    Val path:   {cfg_data.get('val')}")
                print(f"    Test path:  {cfg_data.get('test')}")
    else:
        print("  [!] No .yaml configuration files detected.")

    print("\n--- Split Warning & Sanity Check ---")
    has_train = "train" in stats["splits_found"] and stats["splits_found"]["train"]["images"] > 0
    has_val = ("valid" in stats["splits_found"] and stats["splits_found"]["valid"]["images"] > 0) or \
              ("val" in stats["splits_found"] and stats["splits_found"]["val"]["images"] > 0)
    has_test = "test" in stats["splits_found"] and stats["splits_found"]["test"]["images"] > 0

    if has_train and not has_val and not has_test:
        print("  [WARNING] The dataset only contains training images (0 validation, 0 test).")
        print("  You MUST establish an independent validation split before evaluating or training.")
    elif stats["total_images"] == 0:
        print("  [INFO] No images detected. Please extract the Roboflow dataset into this folder.")
    else:
        print(f"  Status: Train: {has_train}, Val: {has_val}, Test: {has_test}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
