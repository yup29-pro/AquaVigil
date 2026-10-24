# AquaVigil Model Evaluation Protocol & Rules

## 1. Split Discipline & Anti-Leakage Rules
- **Training and Tuning**: All training iterations, hyperparameter tuning, loss weighting, and confidence threshold searches must be conducted strictly using the `train` and `valid` splits.
- **Test Set Quarantine**: The `test` split is strictly held out and quarantined.
- **Single Test Evaluation**: The test split must be evaluated **exactly once** at the very conclusion of Phase 4/5 after model architectures and weights are finalized. Evaluating the test set iteratively to make modeling decisions is prohibited.
- **Safeguard Enforcement**: `training/evaluate.py` will refuse to evaluate the `test` split unless explicitly passed the `--allow-test` flag.

## 2. Standard Evaluation Metrics
Every evaluation run must report:
1. **Precision (P)**: Overall and per-class.
2. **Recall (R)**: Overall and per-class.
3. **mAP@50**: Mean Average Precision at IoU threshold 0.50 overall and per-class.
4. **mAP@50-95**: Mean Average Precision across IoU thresholds 0.50:0.95 overall and per-class.
5. **Confusion Matrix**: Saved visually and serialized to JSON to track misclassifications between background and disease categories.
6. **Inference Latency**: Preprocess, inference, and postprocess latency (in ms per image).

## 3. Class Taxonomy (baseline_v1)
- `0: Aeromoniasis`
- `1: branchitis`
- `2: keratitis`
- `3: lymphocystis`
- `4: object` (reference / foreground tag)

## 4. Evaluation CLI Usage
```bash
# Validate checkpoint on validation set
python training/evaluate.py --weights runs/baseline_v1_yolov8n/weights/best.pt --split val

# Final benchmark on test set (RUN ONCE ONLY)
python training/evaluate.py --weights runs/baseline_v1_yolov8n/weights/best.pt --split test --allow-test
```
