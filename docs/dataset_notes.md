# AquaVigil Dataset Notes & Verification Log

**Verification Date**: 2026-10-21  
**Status**: Verified from raw data and processed splits (`split_v1` and `baseline_v1`).

---

## 1. Provenance & Metadata

- **Dataset Source**: Roboflow Universe (`fish disease - v1`)
- **Roboflow Workspace**: `yashwanth-r`
- **Roboflow Project**: `fish-disease-fqkyc-rh1rk` (Version 1)
- **License**: CC BY 4.0
- **URL**: [https://app.roboflow.com/yashwanth-r/fish-disease-fqkyc-rh1rk/1](https://app.roboflow.com/yashwanth-r/fish-disease-fqkyc-rh1rk/1)
- **Total Raw Images**: 9,011 images (100% `.jpg` format)
- **Raw Packaging**: 9,011 images bundled in `train/` with 0 validation and 0 test images.

---

## 2. Split Strategy & Leakage Prevention Methodology

To convert the monolithic raw dataset into an unbiased, leakage-safe dataset suitable for object detection:

1. **Source Stem & Augmentation Grouping**:
   - Filenames were analyzed and grouped by underlying source identity.
   - Augmentation prefixes (`aug_box_xflip_`, `aug_box_yflip_`, `aug_origin_`, `aug_xflip_`, `aug_yflip_`) and `.rf.<hash>` suffixes were stripped.
   - All augmented variants belonging to a source image are strictly grouped together.
2. **Video-Level Grouping**:
   - All 1,398 video frames were mapped to their 8 specific **Video IDs** (`-new_fish11`, `-new_fish13`, `-new_fish17`, `-new_fish18`, `-new_fish2`, `-new_fish4`, `-new_fish7`, `fish24`).
   - No video was divided across splits; each video was assigned as an indivisible unit to avoid frame leakage between near-identical adjacent frames.
3. **Union-Find Duplicate Merging**:
   - 115 bitwise duplicate pairs detected via MD5 hash comparison were unified into identical group IDs so that identical scenes never straddle split boundaries.
4. **Evaluation Purity Rule**:
   - Validation and test splits contain **only original, non-augmented images**.
   - Augmented copies (`aug_xflip`, `aug_yflip`, etc.) of validation and test groups were excluded from all splits (2,407 augmented images excluded). Train may contain augmented images.
5. **Noisy Label & Malformed Box Cleaning**:
   - The single image with noisy label `1` (`aug_box_xflip_105_jpg.rf.322aa06034d5af5969d1c48b43dc4762.jpg`) was excluded from processed copies.
   - 1 malformed box with $h=0.0$ in `130_fish24_vid-frame_jpg.rf.608ad7691330791fb91b804cbb8c1521.txt` was dropped during copy.

---

## 3. Dataset Versions & Verified Counts

### A. Full Split (`dataset/processed/split_v1`)
Preserves all 9 original class definitions (IDs 0–8) without remapping:
- **`split_v1/data.yaml`**: `nc: 9`, classes `['1', 'Aeromoniasis', 'branchitis', 'healthy', 'ichthyophthiriasis', 'keratitis', 'lymphocystis', 'object', 'saprolegniasis']`.
- **Train**: 5,803 images | 10,191 valid boxes | 259 background (empty label) images
- **Valid**: 400 images (13.4% of originals) | 596 valid boxes | 6 background images
- **Test**: 400 images (13.4% of originals) | 635 valid boxes | 5 background images

### B. Clean Baseline (`dataset/processed/baseline_v1`)
A 5-class benchmark dataset derived from `split_v1` by keeping only the core disease classes and object labels.
- **`baseline_v1/data.yaml`**: `nc: 5`, classes `['Aeromoniasis', 'branchitis', 'keratitis', 'lymphocystis', 'object']`.
- **Entire Image Exclusion**: Any image containing `healthy`, `ichthyophthiriasis`, `saprolegniasis`, or class `1` was completely excluded (116 images total excluded: 110 Train, 4 Val, 2 Test).
- **Remapped Class IDs (0–4)**:
  - `0`: `Aeromoniasis`
  - `1`: `branchitis`
  - `2`: `keratitis`
  - `3`: `lymphocystis`
  - `4`: `object`

| Split | Images Scanned | Labels Matched | Empty Labels | Total Boxes (0–4) | Malformed Lines |
|---|---|---|---|---|---|
| **Train** | **5,693** | 5,693 | 259 | **8,990** | 0 |
| **Valid** | **396** | 396 | 6 | **590** | 0 |
| **Test** | **398** | 398 | 5 | **630** | 0 |
| **Total** | **6,487** | **6,487** | **270** | **10,210** | **0** |

#### Baseline Class Breakdown:
| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes |
|---|---|---|---|---|---|
| 0 | `Aeromoniasis` | 4,087 | 234 | 207 | 4,528 |
| 1 | `branchitis` | 1,023 | 66 | 84 | 1,173 |
| 2 | `keratitis` | 1,205 | 54 | 93 | 1,352 |
| 3 | `lymphocystis` | 1,149 | 57 | 99 | 1,305 |
| 4 | `object` | 1,526 | 179 | 147 | 1,852 |

---

## 4. Verified Limitations & Dataset Insights

1. **`healthy` Class Video-Isolation Limitation**:
   - The `healthy` class (1,189 boxes across 101 frames) exists **exclusively inside a single video (`-new_fish2`)**.
   - Because video frames cannot be split across sets without severe frame-leakage, `healthy` cannot be reliably evaluated and risks learning video-specific visual features. It is excluded from `baseline_v1`.
2. **Rare Classes**:
   - `saprolegniasis` (3 boxes), `ichthyophthiriasis` (20 boxes), and class `1` (1 box) are too sparse to support balanced training or meaningful evaluation; excluded from `baseline_v1`.
3. **`object` Class Semantics**:
   - The meaning of `object` (1,852 boxes) represents general fish body, fins, or non-symptom visual structures rather than a specific pathology, pending manual review.
4. **Source Asymmetry for `object` in Val/Test**:
   - In Validation and Test splits, all disease classes (`Aeromoniasis`, `branchitis`, `keratitis`, `lymphocystis`) originate from static image captures.
   - The `object` class boxes in Val and Test come from one video each (`-new_fish7` in Validation, `-new_fish17` in Test).
5. **Evaluation Purity**:
   - Validation and test splits contain exclusively non-augmented original images to ensure benchmark integrity. Offline synthetic augmentations exist only in the training set.
