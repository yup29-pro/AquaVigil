#!/usr/bin/env python3
"""
scripts/validate_dataset.py

Performs comprehensive validation of a YOLO-formatted dataset:
- Validates data.yaml configuration and class definitions
- Checks image/label 1-to-1 correspondence
- Identifies missing label files, empty annotations, and malformed lines
- Validates coordinate bounds [0.0, 1.0] for YOLO axis-aligned boxes
- Validates class IDs against the defined class list
- Detects duplicate files using MD5 hashing (optional check)
- Generates a readable validation summary

Usage:
    python scripts/validate_dataset.py --config dataset/raw/data.yaml
    python scripts/validate_dataset.py --data-dir dataset/raw
"""

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_yolo_label_line(line: str, line_no: int, num_classes: int) -> tuple[bool, str]:
    parts = line.strip().split()
    if not parts:
        return True, ""  # Empty line, ignored

    if len(parts) != 5:
        return False, f"Line {line_no}: Expected 5 elements (class x_center y_center width height), got {len(parts)}: '{line.strip()}'"

    # Class ID validation
    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"Line {line_no}: Class ID '{parts[0]}' is not an integer"

    if num_classes is not None and (class_id < 0 or class_id >= num_classes):
        return False, f"Line {line_no}: Class ID {class_id} out of bounds (allowed: 0 to {num_classes - 1})"

    # Coordinates validation
    coords = []
    coord_names = ["x_center", "y_center", "width", "height"]
    for i, p in enumerate(parts[1:], start=0):
        try:
            val = float(p)
            coords.append(val)
        except ValueError:
            return False, f"Line {line_no}: Coordinate '{p}' for {coord_names[i]} is not a valid float"

    for i, val in enumerate(coords):
        if val < 0.0 or val > 1.0:
            return False, f"Line {line_no}: Coordinate {coord_names[i]} = {val} outside normalized [0.0, 1.0] range"

    # Box width/height check
    if coords[2] <= 0.0 or coords[3] <= 0.0:
        return False, f"Line {line_no}: Bounding box has non-positive width/height (w={coords[2]}, h={coords[3]})"

    return True, ""


def validate_split(image_dir: Path, label_dir: Path, num_classes: int, check_duplicates: bool = False) -> dict:
    results = {
        "image_count": 0,
        "label_count": 0,
        "missing_labels": [],
        "empty_labels": [],
        "malformed_labels": {},
        "valid_annotations": 0,
        "duplicates": [],
    }

    if not image_dir.exists():
        return results

    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    results["image_count"] = len(images)

    hashes = {}
    for img_path in images:
        if check_duplicates:
            h = compute_file_hash(img_path)
            if h in hashes:
                results["duplicates"].append((img_path.name, hashes[h]))
            else:
                hashes[h] = img_path.name

        expected_label = label_dir / f"{img_path.stem}.txt"
        if not expected_label.exists():
            results["missing_labels"].append(img_path.name)
            continue

        results["label_count"] += 1
        try:
            with open(expected_label, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
        except Exception as e:
            results["malformed_labels"][expected_label.name] = [f"Could not read label file: {e}"]
            continue

        non_empty_lines = [l.strip() for l in lines if l.strip()]
        if not non_empty_lines:
            results["empty_labels"].append(expected_label.name)
            continue

        file_errors = []
        for line_no, line in enumerate(lines, start=1):
            valid, err = validate_yolo_label_line(line, line_no, num_classes)
            if not valid:
                file_errors.append(err)
            elif line.strip():
                results["valid_annotations"] += 1

        if file_errors:
            results["malformed_labels"][expected_label.name] = file_errors

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure, labels, and classes.")
    parser.add_argument("--config", type=str, default=None, help="Path to data.yaml configuration file")
    parser.add_argument("--data-dir", type=str, default="dataset/raw", help="Path to raw dataset directory")
    parser.add_argument("--check-duplicates", action="store_true", help="Enable MD5 hashing to check for duplicate images")
    args = parser.parse_args()

    print("=" * 70)
    print(" AquaVigil - Dataset Validator")
    print("=" * 70)

    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        candidate = Path(args.data_dir) / "data.yaml"
        if candidate.exists():
            config_path = candidate

    class_names = []
    num_classes = None

    if config_path and config_path.exists():
        print(f"Loading Configuration: {config_path.resolve()}")
        if yaml is None:
            print("[!] Warning: PyYAML is not installed. Run 'pip install PyYAML'.")
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                class_names = cfg.get("names", [])
                if isinstance(class_names, list):
                    num_classes = len(class_names)
                elif isinstance(class_names, dict):
                    num_classes = len(class_names.keys())
                    class_names = [class_names[k] for k in sorted(class_names.keys())]
                print(f"Configured Classes ({num_classes}): {class_names}")
            except Exception as e:
                print(f"[!] Error parsing YAML config: {e}")
    else:
        print("[!] Note: No data.yaml found or specified. Validation will skip class ID bound verification.")

    base_dir = Path(args.data_dir)
    splits_to_check = ["train", "valid", "val", "test"]
    found_any_split = False

    for split in splits_to_check:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"

        # Alternative structure fallback: split directly contains images and labels
        if not img_dir.exists() and (base_dir / split).exists():
            img_dir = base_dir / split
            lbl_dir = base_dir / split

        if img_dir.exists():
            found_any_split = True
            print(f"\n--- Validating Split: {split.upper()} ---")
            print(f"  Images Path: {img_dir}")
            print(f"  Labels Path: {lbl_dir}")

            res = validate_split(img_dir, lbl_dir, num_classes, args.check_duplicates)
            print(f"  Images Scanned:       {res['image_count']}")
            print(f"  Labels Matched:       {res['label_count']}")
            print(f"  Valid Annotations:    {res['valid_annotations']}")
            print(f"  Missing Label Files:  {len(res['missing_labels'])}")
            print(f"  Empty Label Files:    {len(res['empty_labels'])}")
            print(f"  Malformed Label Files:{len(res['malformed_labels'])}")

            if res["missing_labels"]:
                print(f"    Sample missing: {res['missing_labels'][:5]}")
            if res["empty_labels"]:
                print(f"    Sample empty (background/no-disease images): {res['empty_labels'][:5]}")
            if res["malformed_labels"]:
                print(f"    [!] Error sample in malformed files:")
                for fname, errs in list(res["malformed_labels"].items())[:3]:
                    print(f"      - {fname}: {errs[0]}")
            if args.check_duplicates and res["duplicates"]:
                print(f"    Duplicate images found: {len(res['duplicates'])}")
                for dup, orig in res["duplicates"][:3]:
                    print(f"      - {dup} is identical to {orig}")

    if not found_any_split:
        print(f"\n[!] No standard split directories found in {base_dir.resolve()}.")
        print("Please verify that the dataset is extracted into dataset/raw/.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
