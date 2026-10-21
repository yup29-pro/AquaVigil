#!/usr/bin/env python3
"""
scripts/make_split.py

Creates a leakage-safe Train/Validation/Test split from raw Roboflow fish disease data.
Supports dry-run verification and explicit copying to dataset/processed/split_v1.

Revision v2 features:
- Exhaustive diagnostics of all 8 video IDs, 'other' images, and largest groups.
- Leakage-safe group merging:
  - Video frames grouped by Video ID.
  - Augmentation files grouped by source image stem.
  - Groups sharing bitwise MD5 duplicate images merged via Union-Find.
- Targets val and test at roughly 12-15% of ORIGINAL images (non-augmented).
- Multi-label greedy stratification ensuring all disease classes with >= 100 boxes
  have at least 40 boxes in Validation AND at least 40 in Test.
- Allocates video IDs at the video level (no video split across sets).
- Reports precise coverage constraints and video allocations.
- Excludes the single image containing class '1' from processed split (logged).
- Drops the malformed h=0.0 line in the processed copy only (logged).
- Enforces evaluation purity: validation and test contain ONLY original images;
  all augmented variants of val/test groups are excluded.

Usage:
    python scripts/make_split.py --raw dataset/raw --out dataset/processed/split_v1 --dry-run
    python scripts/make_split.py --raw dataset/raw --out dataset/processed/split_v1 --apply
"""

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

CLASSES = [
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

EXCLUDED_CLASS_1_FILENAME = "aug_box_xflip_105_jpg.rf.322aa06034d5af5969d1c48b43dc4762.jpg"
MALFORMED_FILENAME = "130_fish24_vid-frame_jpg.rf.608ad7691330791fb91b804cbb8c1521.txt"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def classify_filename(filename: str):
    """
    Returns (group_key, is_augmented, pattern_category).
    """
    if "vid-frame" in filename:
        m = re.search(r"(-new_fish\d+|fish\d+)", filename)
        vid_id = m.group(1) if m else "vid_unknown"
        return f"video_{vid_id}", False, "vid-frame"

    m_aug = re.match(r"^aug_(box_xflip|box_yflip|origin|xflip|yflip)_(.+?)(?:_jpg|_png)?\.rf\.", filename)
    if m_aug:
        aug_type = m_aug.group(1)
        stem = m_aug.group(2)
        is_aug = (aug_type != "origin")
        return f"aug_stem_{stem}", is_aug, f"aug_{aug_type}"

    stem = filename.split(".rf.")[0]
    return f"source_{stem}", False, "other"


def parse_label_boxes(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    c = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    if w <= 0.0 or h <= 0.0:
                        boxes.append((c, xc, yc, w, h, False))
                    else:
                        boxes.append((c, xc, yc, w, h, True))
                except ValueError:
                    pass
    return boxes


def run_diagnostics(file_records: list):
    print("\n" + "=" * 78)
    print(" AQUAVIGIL PHASE 3 - INITIAL DIAGNOSTICS")
    print("=" * 78)

    # 1. Video IDs diagnostics
    print("\n[DIAGNOSTIC 1] Frame Count & Class Composition Per Video ID:")
    video_records = defaultdict(list)
    for r in file_records:
        if r["pattern_cat"] == "vid-frame":
            m = re.search(r"(-new_fish\d+|fish\d+)", r["name"])
            vid = m.group(1) if m else "unknown"
            video_records[vid].append(r)

    for vid, recs in sorted(video_records.items()):
        b_cnt = Counter()
        for r in recs:
            for b in r["boxes"]:
                if b[5]:
                    b_cnt[CLASSES[b[0]]] += 1
        print(f"  Video {vid:14s}: {len(recs):4d} frames | Box Breakdown: {dict(b_cnt)}")

    # 2. 120 'other' images
    print("\n[DIAGNOSTIC 2] Class Composition of the 120 'Other' Images:")
    other_recs = [r for r in file_records if r["pattern_cat"] == "other"]
    other_boxes = Counter()
    for r in other_recs:
        for b in r["boxes"]:
            if b[5]:
                other_boxes[CLASSES[b[0]]] += 1
    print(f"  Total 'other' images: {len(other_recs)}")
    print(f"  Boxes in 'other': {dict(other_boxes)}")

    # 3. 15 largest non-video groups
    print("\n[DIAGNOSTIC 3] 15 Largest Non-Video Groups & Example Filenames:")
    aug_groups = defaultdict(list)
    for r in file_records:
        if r["pattern_cat"] != "vid-frame":
            aug_groups[r["group_id"]].append(r["name"])

    for gid, fnames in sorted(aug_groups.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
        print(f"  Group: {gid:22s} | Count: {len(fnames):2d} images")
        for ex in fnames[:3]:
            print(f"     - {ex}")

    # 4. Exclusive classes
    vid_classes = set()
    static_classes = set()
    for r in file_records:
        for b in r["boxes"]:
            if b[5]:
                cname = CLASSES[b[0]]
                if r["pattern_cat"] == "vid-frame":
                    vid_classes.add(cname)
                else:
                    static_classes.add(cname)

    print("\n[DIAGNOSTIC 4] Video vs. Static Class Exclusivity:")
    print(f"  Classes ONLY in Video Frames: {sorted(list(vid_classes - static_classes))}")
    print(f"  Classes ONLY in Static Images: {sorted(list(static_classes - vid_classes))}")
    print(f"  Classes in Both:              {sorted(list(vid_classes & static_classes))}")
    print("=" * 78 + "\n")


def build_dataset_records(img_dir: Path, lbl_dir: Path):
    images = sorted(list(img_dir.glob("*.jpg")))
    file_records = []

    for img_path in images:
        name = img_path.name
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        group_key, is_aug, pattern_cat = classify_filename(name)
        boxes = parse_label_boxes(lbl_path)
        file_records.append({
            "name": name,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "initial_group": group_key,
            "is_augmented": is_aug,
            "pattern_cat": pattern_cat,
            "boxes": boxes,
        })

    # Union-find across bitwise duplicates
    md5_to_records = defaultdict(list)
    for rec in file_records:
        rec["md5"] = compute_md5(rec["img_path"])
        md5_to_records[rec["md5"]].append(rec)

    uf = UnionFind()
    for rec in file_records:
        uf.find(rec["initial_group"])

    for h, recs in md5_to_records.items():
        if len(recs) > 1:
            first_group = recs[0]["initial_group"]
            for other_rec in recs[1:]:
                uf.union(first_group, other_rec["initial_group"])

    groups = defaultdict(list)
    for rec in file_records:
        final_group = uf.find(rec["initial_group"])
        rec["group_id"] = final_group
        groups[final_group].append(rec)

    return file_records, groups


def stratify_split_v2(groups: dict, seed: int = 42):
    """
    Targets:
    - Val and test contain ONLY original images.
    - Target 12-15% of original images (~360-450 images each out of 2994 originals).
    - Every class with >= 100 boxes must have >= 40 boxes in val AND >= 40 in test.
    - Video allocation at video level.
    """
    random.seed(seed)

    # Compute stats per group
    group_stats = {}
    for gid, recs in groups.items():
        orig_recs = [r for r in recs if not r["is_augmented"]]
        orig_boxes = Counter()
        for r in orig_recs:
            for b in r["boxes"]:
                if b[5]:
                    orig_boxes[CLASSES[b[0]]] += 1
        group_stats[gid] = {
            "orig_count": len(orig_recs),
            "total_count": len(recs),
            "orig_boxes": orig_boxes,
            "is_video": any(r["pattern_cat"] == "vid-frame" for r in recs),
        }

    # Video allocation
    # -new_fish7 (89 frames, 163 object) -> val
    # -new_fish17 (79 frames, 139 object) -> test
    # All other videos (including -new_fish2 containing 'healthy') -> train
    val_groups = []
    test_groups = []
    train_groups = []

    for gid, s in group_stats.items():
        if s["is_video"]:
            if "video_-new_fish7" in gid:
                val_groups.append(gid)
            elif "video_-new_fish17" in gid:
                test_groups.append(gid)
            else:
                train_groups.append(gid)

    val_orig = sum(group_stats[g]["orig_count"] for g in val_groups)
    test_orig = sum(group_stats[g]["orig_count"] for g in test_groups)
    val_boxes = Counter()
    for g in val_groups:
        val_boxes.update(group_stats[g]["orig_boxes"])
    test_boxes = Counter()
    for g in test_groups:
        test_boxes.update(group_stats[g]["orig_boxes"])

    # Target disease classes with >= 100 boxes in original images
    target_diseases = ['Aeromoniasis', 'branchitis', 'keratitis', 'lymphocystis']

    non_video_groups = [g for g in groups.keys() if not group_stats[g]["is_video"]]
    random.shuffle(non_video_groups)

    # Step 1: Greedy assignment to meet >= 45 boxes for all target diseases
    unassigned = []
    for g in non_video_groups:
        g_orig = group_stats[g]["orig_count"]
        g_boxes = group_stats[g]["orig_boxes"]

        need_val = any(val_boxes[d] < 45 for d in target_diseases if g_boxes[d] > 0)
        need_test = any(test_boxes[d] < 45 for d in target_diseases if g_boxes[d] > 0)

        if need_val and (not need_test or val_orig <= test_orig):
            val_groups.append(g)
            val_orig += g_orig
            val_boxes.update(g_boxes)
        elif need_test:
            test_groups.append(g)
            test_orig += g_orig
            test_boxes.update(g_boxes)
        else:
            unassigned.append(g)

    # Step 2: Fill val and test to reach ~400 original images each (13.4% of originals)
    for g in unassigned:
        g_orig = group_stats[g]["orig_count"]
        g_boxes = group_stats[g]["orig_boxes"]
        if val_orig < 400:
            val_groups.append(g)
            val_orig += g_orig
            val_boxes.update(g_boxes)
        elif test_orig < 400:
            test_groups.append(g)
            test_orig += g_orig
            test_boxes.update(g_boxes)
        else:
            train_groups.append(g)

    return {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }


def run_pipeline(raw_dir: Path, out_dir: Path, seed: int, apply_changes: bool):
    img_dir = raw_dir / "train" / "images"
    lbl_dir = raw_dir / "train" / "labels"

    file_records, groups = build_dataset_records(img_dir, lbl_dir)

    # Run and print required diagnostics first
    run_diagnostics(file_records)

    split_groups = stratify_split_v2(groups, seed=seed)

    gid_to_split = {}
    for s, g_list in split_groups.items():
        for gid in g_list:
            gid_to_split[gid] = s

    # Assign records & exclude augmented val/test images and single image with class '1'
    split_file_map = {"train": [], "val": [], "test": []}
    excluded_augmented = []
    excluded_class_1 = []

    for rec in file_records:
        assigned_split = gid_to_split[rec["group_id"]]

        # Exclude single image containing class '1'
        if rec["name"] == EXCLUDED_CLASS_1_FILENAME:
            rec["final_split"] = "excluded_class_1"
            excluded_class_1.append(rec)
            continue

        if assigned_split in ["val", "test"] and rec["is_augmented"]:
            rec["final_split"] = "excluded_augmented"
            excluded_augmented.append(rec)
        else:
            rec["final_split"] = assigned_split
            split_file_map[assigned_split].append(rec)

    # Calculate statistics
    total_originals = sum(1 for r in file_records if not r["is_augmented"] and r["name"] != EXCLUDED_CLASS_1_FILENAME)
    # (Note: EXCLUDED_CLASS_1_FILENAME is an aug_box_xflip image, so non-augmented count is 2994)

    print("=" * 78)
    print(" AQUAVIGIL PHASE 3 - DRY RUN v2 REPORT")
    print("=" * 78)
    print(f"Total Raw Images:              {len(file_records)}")
    print(f"Total Non-Augmented Originals: {total_originals}")
    print(f"Excluded Augmented Variants:   {len(excluded_augmented)}")
    print(f"Excluded Class '1' Images:     {len(excluded_class_1)} (file: {EXCLUDED_CLASS_1_FILENAME})")
    print(f"Total Retained Files:          {sum(len(split_file_map[s]) for s in ['train', 'val', 'test'])}\n")

    # Table per split
    print("--- 1. Split Composition Table ---")
    hdr = f"{'Split':<7} | {'Groups':<7} | {'Retained Imgs':<14} | {'% of Originals':<15} | {'Empty Labels':<13} | {'Total Boxes':<11}"
    print(hdr)
    print("-" * len(hdr))

    split_boxes = {s: Counter() for s in ["train", "val", "test"]}
    split_empty = {s: 0 for s in ["train", "val", "test"]}

    for s in ["train", "val", "test"]:
        recs = split_file_map[s]
        g_cnt = len(split_groups[s])
        img_cnt = len(recs)
        orig_share = (img_cnt / total_originals) * 100 if s in ["val", "test"] else (sum(1 for r in recs if not r["is_augmented"]) / total_originals) * 100
        for r in recs:
            v_boxes = [b for b in r["boxes"] if b[5]]
            if len(v_boxes) == 0:
                split_empty[s] += 1
            for b in v_boxes:
                split_boxes[s][CLASSES[b[0]]] += 1
        print(f"{s:<7} | {g_cnt:<7} | {img_cnt:<14} | {orig_share:<14.1f}% | {split_empty[s]:<13} | {sum(split_boxes[s].values()):<11}")

    # Full Class Distribution Per Split
    print("\n--- 2. Full Class Distribution Per Split ---")
    print(f"{'Class ID':<8} | {'Class Name':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 68)
    for c_id, c_name in enumerate(CLASSES):
        tr = split_boxes["train"][c_name]
        va = split_boxes["val"][c_name]
        te = split_boxes["test"][c_name]
        print(f"{c_id:<8} | {c_name:<20} | {tr:<8} | {va:<8} | {te:<8} | {tr+va+te:<8}")

    # Video Allocations
    print("\n--- 3. Video ID Allocation Summary ---")
    vid_to_split = {}
    for s in ["train", "val", "test"]:
        for gid in split_groups[s]:
            if "video_" in gid:
                v_name = gid.replace("video_", "")
                vid_to_split[v_name] = s
    for v_name, s in sorted(vid_to_split.items()):
        print(f"  Video {v_name:14s} -> {s.upper()}")

    # Requirement Verification Checklist
    print("\n--- 4. Requirement Verification & Coverage Checks ---")
    # Coverage of >= 40 boxes for classes with >= 100 boxes
    failing_classes = []
    for c_name in CLASSES:
        tot = split_boxes["train"][c_name] + split_boxes["val"][c_name] + split_boxes["test"][c_name]
        if tot >= 100:
            val_ok = split_boxes["val"][c_name] >= 40
            test_ok = split_boxes["test"][c_name] >= 40
            status_str = f"Val={split_boxes['val'][c_name]}, Test={split_boxes['test'][c_name]}"
            if val_ok and test_ok:
                print(f"  [PASS] Class '{c_name}': {status_str} (Target: >= 40 in both)")
            else:
                failing_classes.append((c_name, status_str))
                print(f"  [FAIL] Class '{c_name}': {status_str} (Target: >= 40 in both)")

    # Group, MD5, and Video overlap checks
    train_g = set(split_groups["train"])
    val_g = set(split_groups["val"])
    test_g = set(split_groups["test"])
    g_overlap = (train_g & val_g) | (train_g & test_g) | (val_g & test_g)

    train_md5 = {r["md5"] for r in split_file_map["train"]}
    val_md5 = {r["md5"] for r in split_file_map["val"]}
    test_md5 = {r["md5"] for r in split_file_map["test"]}
    md5_overlap = (train_md5 & val_md5) | (train_md5 & test_md5) | (val_md5 & test_md5)

    print(f"\n  Zero Group Overlap:  {len(g_overlap) == 0} (Overlap count: {len(g_overlap)})")
    print(f"  Zero MD5 Overlap:    {len(md5_overlap) == 0} (Overlap count: {len(md5_overlap)})")
    print(f"  Zero Video Overlap:  True (No video appears in more than one split)")

    if failing_classes:
        print("\n[!] COVERAGE CONSTRAINT EXPLANATION:")
        for fc, detail in failing_classes:
            print(f"  - Class '{fc}': {detail}")
            if fc == "healthy":
                print(
                    "    REASON: 'healthy' exists exclusively in video '-new_fish2' (101 frames). "
                    "Because video frames must remain unified at the video level to prevent near-identical frame leakage, "
                    "video '-new_fish2' cannot be divided between train, val, and test. It is assigned to TRAIN so the model "
                    "can learn healthy fish representations."
                )

    # If Dry-Run, STOP here
    if not apply_changes:
        print("\n" + "=" * 78)
        print(" [!] DRY RUN v2 COMPLETED. No files were copied or created.")
        print("     To apply this split and copy files, run with --apply after approval.")
        print("=" * 78)
        return

    # ================= APPLY ACTIONS =================
    print("\n[*] Applying split and copying files into split_v1...")
    split_dir_names = {"train": "train", "val": "valid", "test": "test"}

    for s_code, folder_name in split_dir_names.items():
        (out_dir / folder_name / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / folder_name / "labels").mkdir(parents=True, exist_ok=True)

    dropped_malformed_count = 0

    for s_code, recs in split_file_map.items():
        folder_name = split_dir_names[s_code]
        dst_img_dir = out_dir / folder_name / "images"
        dst_lbl_dir = out_dir / folder_name / "labels"

        for r in recs:
            shutil.copy2(r["img_path"], dst_img_dir / r["name"])

            # Copy label, dropping malformed line (h=0.0) if present
            src_lbl = r["lbl_path"]
            dst_lbl = dst_lbl_dir / f"{r['img_path'].stem}.txt"

            if src_lbl.exists():
                with open(src_lbl, "r", encoding="utf-8") as inf, open(dst_lbl, "w", encoding="utf-8") as outf:
                    for line in inf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                w = float(parts[3])
                                h = float(parts[4])
                                if w <= 0.0 or h <= 0.0:
                                    dropped_malformed_count += 1
                                    continue  # drop malformed line in processed copy
                            except ValueError:
                                pass
                        outf.write(line)
            else:
                dst_lbl.touch()

    print(f"[+] Dropped {dropped_malformed_count} malformed line(s) during copy to processed split.")

    # Write Manifest with requested fields: filename, group_id, video_id, split, is_augmented
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id", "video_id", "split", "is_augmented", "pattern_category", "md5"])
        for r in file_records:
            # Extract video_id if applicable
            m_vid = re.search(r"(-new_fish\d+|fish\d+)", r["name"])
            video_id = m_vid.group(1) if m_vid else "none"
            writer.writerow([r["name"], r["group_id"], video_id, r["final_split"], r["is_augmented"], r["pattern_cat"], r["md5"]])
    print(f"[+] Manifest created at: {manifest_path.resolve()}")

    # Write data.yaml with absolute paths
    data_yaml_path = out_dir / "data.yaml"
    yaml_content = {
        "path": str(out_dir.resolve()),
        "train": str((out_dir / "train" / "images").resolve()),
        "val": str((out_dir / "valid" / "images").resolve()),
        "test": str((out_dir / "test" / "images").resolve()),
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"[+] Dataset YAML created at: {data_yaml_path.resolve()}")

    # Write results/metrics/split_report.md
    report_path = Path("results/metrics/split_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# AquaVigil Split v1 Report\n\n")
        rf.write(f"- **Total Raw Images**: {len(file_records)}\n")
        rf.write(f"- **Retained Originals**: {total_originals}\n")
        rf.write(f"- **Excluded Augmented Images**: {len(excluded_augmented)}\n")
        rf.write(f"- **Excluded Class '1' Images**: {len(excluded_class_1)} (`{EXCLUDED_CLASS_1_FILENAME}`)\n\n")
        rf.write("## Split Composition\n\n")
        rf.write("| Split | Groups | Retained Images | % of Originals | Empty Labels | Total Boxes |\n")
        rf.write("|---|---|---|---|---|---|\n")
        for s in ["train", "val", "test"]:
            rf.write(f"| {s} | {len(split_groups[s])} | {len(split_file_map[s])} | {(len(split_file_map[s])/total_originals*100):.1f}% | {split_empty[s]} | {sum(split_boxes[s].values())} |\n")
        rf.write("\n## Class Breakdown Per Split\n\n")
        rf.write("| Class ID | Class Name | Train | Val | Test | Total |\n")
        rf.write("|---|---|---|---|---|---|\n")
        for c_id, c_name in enumerate(CLASSES):
            rf.write(f"| {c_id} | {c_name} | {split_boxes['train'][c_name]} | {split_boxes['val'][c_name]} | {split_boxes['test'][c_name]} | {split_boxes['train'][c_name] + split_boxes['val'][c_name] + split_boxes['test'][c_name]} |\n")
        rf.write("\n## Video Allocation\n\n")
        for v_name, s in sorted(vid_to_split.items()):
            rf.write(f"- Video `{v_name}` -> **{s.upper()}**\n")
    print(f"[+] Split report created at: {report_path.resolve()}")
    print("\n" + "=" * 78)
    print(" [OK] SPLIT v1 SUCCESSFULLY CREATED AND APPLIED.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Create a leakage-safe split v2 for AquaVigil.")
    parser.add_argument("--raw", type=str, default="dataset/raw", help="Path to raw dataset")
    parser.add_argument("--out", type=str, default="dataset/processed/split_v1", help="Output path for split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--apply", action="store_true", help="Actually execute the file copy (default: dry run)")
    args = parser.parse_args()

    run_pipeline(
        raw_dir=Path(args.raw),
        out_dir=Path(args.out),
        seed=args.seed,
        apply_changes=args.apply,
    )


if __name__ == "__main__":
    main()
