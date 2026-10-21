#!/usr/bin/env python3
"""
scripts/make_baseline.py

Creates the 5-class baseline dataset (baseline_v1) derived strictly from split_v1.

Rules:
- Keep classes: Aeromoniasis, branchitis, keratitis, lymphocystis, object.
- Remap to contiguous IDs 0-4 in this order:
  0: Aeromoniasis
  1: branchitis
  2: keratitis
  3: lymphocystis
  4: object
- EXCLUDE ENTIRE IMAGES (and their labels) that contain any of:
  healthy, ichthyophthiriasis, saprolegniasis, class "1".
  (Do NOT just delete those boxes; remove the whole image from the dataset).
- Preserve the exact same split assignment as split_v1 (no re-splitting).
- Writes baseline_v1/{train,valid,test}/{images,labels}
- Writes baseline_v1/data.yaml (absolute paths, nc=5, names list)
- Writes baseline_v1/manifest.csv

Usage:
    python scripts/make_baseline.py --src dataset/processed/split_v1 --out dataset/processed/baseline_v1 --dry-run
    python scripts/make_baseline.py --src dataset/processed/split_v1 --out dataset/processed/baseline_v1 --apply
"""

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

RAW_CLASSES = [
    '1',
    'Aeromoniasis',
    'branchitis',
    'healthy',
    'ichthyophthiriasis',
    'keratitis',
    'lymphocystis',
    'object',
    'saprolegniasis',
]

BASELINE_CLASSES = [
    'Aeromoniasis',
    'branchitis',
    'keratitis',
    'lymphocystis',
    'object',
]

# Raw IDs for excluded classes: 0 ('1'), 3 ('healthy'), 4 ('ichthyophthiriasis'), 8 ('saprolegniasis')
EXCLUDED_RAW_IDS = {0, 3, 4, 8}

# Mapping raw class ID -> baseline class ID (0 to 4)
CLASS_REMAP = {
    1: 0,  # Aeromoniasis
    2: 1,  # branchitis
    5: 2,  # keratitis
    6: 3,  # lymphocystis
    7: 4,  # object
}


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_split_v1(src_dir: Path):
    split_folders = {"train": "train", "val": "valid", "test": "test"}
    records = []

    for s_code, folder_name in split_folders.items():
        img_dir = src_dir / folder_name / "images"
        lbl_dir = src_dir / folder_name / "labels"

        if not img_dir.exists():
            continue

        for img_path in sorted(list(img_dir.glob("*.jpg"))):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw_boxes = []
            has_excluded_class = False
            excluded_reasons = set()

            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                raw_cls = int(parts[0])
                                xc = float(parts[1])
                                yc = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                if raw_cls in EXCLUDED_RAW_IDS:
                                    has_excluded_class = True
                                    excluded_reasons.add(RAW_CLASSES[raw_cls])
                                raw_boxes.append((raw_cls, xc, yc, w, h))
                            except ValueError:
                                pass

            records.append({
                "name": img_path.name,
                "img_path": img_path,
                "lbl_path": lbl_path,
                "split": s_code,
                "folder_name": folder_name,
                "raw_boxes": raw_boxes,
                "has_excluded_class": has_excluded_class,
                "excluded_reasons": list(excluded_reasons),
            })

    return records


def run_baseline_pipeline(src_dir: Path, out_dir: Path, apply_changes: bool):
    print("=" * 78)
    print(" AQUAVIGIL BASELINE v1 BUILDER (5 CLASSES)")
    print("=" * 78)
    print(f"Source Directory: {src_dir.resolve()}")
    print(f"Output Directory: {out_dir.resolve()}")
    print(f"Execution Mode:   {'*** APPLY (WRITE FILES) ***' if apply_changes else 'DRY RUN (NO FILES TOUCHED)'}\n")

    if not src_dir.exists():
        print(f"[!] Error: Source directory does not exist: {src_dir.resolve()}")
        sys.exit(1)

    records = inspect_split_v1(src_dir)
    if not records:
        print(f"[!] Error: No images found in {src_dir.resolve()}. Ensure split_v1 is populated first.")
        sys.exit(1)

    # Classify images into retained vs excluded
    retained_by_split = {"train": [], "val": [], "test": []}
    excluded_by_split = {"train": [], "val": [], "test": []}
    excluded_counts_per_class = Counter()

    for r in records:
        s = r["split"]
        if r["has_excluded_class"]:
            excluded_by_split[s].append(r)
            for reason in r["excluded_reasons"]:
                excluded_counts_per_class[reason] += 1
        else:
            retained_by_split[s].append(r)

    print("--- 1. Excluded Images Summary (Images Containing Excluded Classes) ---")
    print(f"  Total Excluded Images: {sum(len(excluded_by_split[s]) for s in ['train', 'val', 'test'])}")
    for s in ["train", "val", "test"]:
        print(f"    - {s.upper():5s}: {len(excluded_by_split[s]):3d} images excluded")
    print("\n  Dropped Images Breakdown By Class Trigger:")
    for cls_name, cnt in sorted(excluded_counts_per_class.items()):
        print(f"    - Class '{cls_name}': {cnt} image(s) excluded")

    # Metrics per split
    print("\n--- 2. Baseline v1 Split Composition Table ---")
    hdr = f"{'Split':<7} | {'Images Retained':<17} | {'Empty (Background)':<20} | {'Total Boxes (0-4)':<17}"
    print(hdr)
    print("-" * len(hdr))

    split_boxes = {s: Counter() for s in ["train", "val", "test"]}
    split_empty = {s: 0 for s in ["train", "val", "test"]}

    for s in ["train", "val", "test"]:
        recs = retained_by_split[s]
        for r in recs:
            valid_b = [b for b in r["raw_boxes"] if b[0] in CLASS_REMAP]
            if len(valid_b) == 0:
                split_empty[s] += 1
            for b in valid_b:
                remapped_cls = CLASS_REMAP[b[0]]
                split_boxes[s][BASELINE_CLASSES[remapped_cls]] += 1
        print(f"{s:<7} | {len(recs):<17} | {split_empty[s]:<20} | {sum(split_boxes[s].values()):<17}")

    # Full Class Distribution
    print("\n--- 3. Baseline v1 Class Distribution Per Split (Remapped 0-4) ---")
    print(f"{'New ID':<8} | {'Class Name':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 68)
    for new_id, c_name in enumerate(BASELINE_CLASSES):
        tr = split_boxes["train"][c_name]
        va = split_boxes["val"][c_name]
        te = split_boxes["test"][c_name]
        print(f"{new_id:<8} | {c_name:<20} | {tr:<8} | {va:<8} | {te:<8} | {tr+va+te:<8}")

    # Coverage Checks
    print("\n--- 4. Coverage Checks (>= 40 boxes in Val & Test) ---")
    failing = []
    for c_name in BASELINE_CLASSES:
        v_ok = split_boxes["val"][c_name] >= 40
        t_ok = split_boxes["test"][c_name] >= 40
        status_str = f"Val={split_boxes['val'][c_name]}, Test={split_boxes['test'][c_name]}"
        if v_ok and t_ok:
            print(f"  [PASS] Class '{c_name}': {status_str}")
        else:
            failing.append((c_name, status_str))
            print(f"  [FAIL] Class '{c_name}': {status_str}")

    # Leakage Checks
    print("\n--- 5. Leakage & Overlap Verification ---")
    train_names = {r["name"] for r in retained_by_split["train"]}
    val_names = {r["name"] for r in retained_by_split["val"]}
    test_names = {r["name"] for r in retained_by_split["test"]}
    name_overlap = (train_names & val_names) | (train_names & test_names) | (val_names & test_names)
    print(f"  Zero Filename Overlap: {len(name_overlap) == 0} (Overlap: {len(name_overlap)})")

    if not apply_changes:
        print("\n" + "=" * 78)
        print(" [!] DRY RUN COMPLETED. No files were copied or created.")
        print("     To apply baseline_v1, run with --apply after approval.")
        print("=" * 78)
        return

    # Apply changes
    print("\n[*] Applying baseline_v1 and writing remapped dataset...")
    split_folders = {"train": "train", "val": "valid", "test": "test"}
    for s_code, folder_name in split_folders.items():
        (out_dir / folder_name / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / folder_name / "labels").mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for s_code, recs in retained_by_split.items():
        folder_name = split_folders[s_code]
        dst_img_dir = out_dir / folder_name / "images"
        dst_lbl_dir = out_dir / folder_name / "labels"

        for r in recs:
            # Copy image
            shutil.copy2(r["img_path"], dst_img_dir / r["name"])

            # Copy and remap label
            dst_lbl = dst_lbl_dir / f"{r['img_path'].stem}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as outf:
                for b in r["raw_boxes"]:
                    raw_cls = b[0]
                    if raw_cls in CLASS_REMAP:
                        new_cls = CLASS_REMAP[raw_cls]
                        outf.write(f"{new_cls} {b[1]} {b[2]} {b[3]} {b[4]}\n")

            manifest_rows.append([r["name"], s_code, dst_img_dir / r["name"], dst_lbl])

    # Write Manifest
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "image_path", "label_path"])
        writer.writerows(manifest_rows)
    print(f"[+] Manifest created at: {manifest_path.resolve()}")

    # Write data.yaml
    data_yaml_path = out_dir / "data.yaml"
    yaml_content = {
        "path": str(out_dir.resolve()),
        "train": str((out_dir / "train" / "images").resolve()),
        "val": str((out_dir / "valid" / "images").resolve()),
        "test": str((out_dir / "test" / "images").resolve()),
        "nc": len(BASELINE_CLASSES),
        "names": BASELINE_CLASSES,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"[+] Dataset YAML created at: {data_yaml_path.resolve()}")
    print("\n" + "=" * 78)
    print(" [OK] BASELINE v1 DATASET CREATED SUCCESSFULLY.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Build 5-class baseline dataset from split_v1.")
    parser.add_argument("--src", type=str, default="dataset/processed/split_v1", help="Path to split_v1")
    parser.add_argument("--out", type=str, default="dataset/processed/baseline_v1", help="Path to baseline_v1")
    parser.add_argument("--apply", action="store_true", help="Actually execute the copy and remapping")
    args = parser.parse_args()

    run_baseline_pipeline(Path(args.src), Path(args.out), args.apply)


if __name__ == "__main__":
    main()
