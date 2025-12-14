#!/usr/bin/env python3
# scripts/phase1/01_train_teacher_ultimate_optimized_FINAL.py
"""
Minimal, clean YOLO11 teacher training script that:
 - Loads best HPO params (YAML)
 - Applies them to a minimal Ultralytics / YOLO v11 training config
 - Trains and writes a clean training_summary.json
 - Safe, copy-paste ready
"""
import argparse
import json
from pathlib import Path
import yaml
from datetime import datetime
import sys
from ultralytics import YOLO

# -------------------------
# Utility helpers
# -------------------------
def load_yaml(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _safe_extract_metrics(results) -> dict:
    """
    Robust extraction for common YOLO result shapes (results.results_dict OR results.box.*).
    Returns dict with keys: mAP50, mAP50_95, precision, recall
    """
    m = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    try:
        # 1) Prefer results.results_dict if present (Ultralytics stores metrics as keys)
        rd = getattr(results, "results_dict", None)
        if isinstance(rd, dict):
            # rd keys look like 'metrics/mAP50' or 'metrics/mAP50-95'
            for k, v in rd.items():
                key_lower = str(k).lower()
                if "map50-95" in key_lower or "map50_95" in key_lower:
                    try:
                        m["mAP50_95"] = float(v)
                    except Exception:
                        pass
                elif "map50" in key_lower and "map50-95" not in key_lower and "map50_95" not in key_lower:
                    try:
                        m["mAP50"] = float(v)
                    except Exception:
                        pass
                elif "precision" in key_lower:
                    try:
                        m["precision"] = float(v)
                    except Exception:
                        pass
                elif "recall" in key_lower:
                    try:
                        m["recall"] = float(v)
                    except Exception:
                        pass
        # 2) Fallback to results.box.* attributes (some builds expose results.box.map / map50)
        box = getattr(results, "box", None)
        if box is not None:
            try:
                if getattr(box, "map50", None) is not None:
                    m["mAP50"] = float(box.map50)
            except Exception:
                pass
            try:
                if getattr(box, "map", None) is not None:
                    # many versions expose 'map' as mAP50-95
                    m["mAP50_95"] = float(box.map)
            except Exception:
                pass
        # 3) final numeric cleanup
        for k in ["mAP50", "mAP50_95", "precision", "recall"]:
            if m.get(k) is None:
                m[k] = 0.0
    except Exception as e:
        print(f"⚠️ Warning extracting metrics: {e}")
    return m

def find_best_weights_dir(project: Path, name: str) -> Path:
    """
    Typical ultralytics layout: <project>/<name>/weights/best.pt
    Return the path to best.pt if exists, else return None
    """
    p = project / name / "weights" / "best.pt"
    if p.exists():
        return p
    # sometimes Ultralytics saves directly under project/name as 'best.pt'
    p2 = project / name / "best.pt"
    if p2.exists():
        return p2
    return None

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Minimal YOLO11 teacher training with HPO params")
    p.add_argument("--data", required=True, help="path to data.yaml (dataset)")
    p.add_argument("--weights", default="yolo11s.pt", help="pretrained weights")
    p.add_argument("--hpo-results",
                   default="runs/hpo_production_final_resumed/final_results/best_params.yaml",
                   help="path to HPO best_params.yaml")
    p.add_argument("--epochs", type=int, default=150, help="training epochs (overrides recommended epoch if provided)")
    p.add_argument("--project", default="runs/teacher_training", help="project folder")
    p.add_argument("--name", default="yolo11s_teacher_final", help="experiment name")
    p.add_argument("--device", default=0, help="device id (0 for first GPU, 'cpu' for CPU)")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--img-size", type=int, default=None, help="override image size")
    return p.parse_args()

# -------------------------
# Main training function
# -------------------------
def main():
    args = parse_args()

    data_yaml = Path(args.data)
    weights = args.weights
    hpo_path = Path(args.hpo_results)
    project = Path(args.project)
    name = args.name

    print("\n" + "="*60)
    print("Minimal YOLO11 Teacher Training (using HPO results when available)")
    print("="*60 + "\n")

    # load hpo params
    hpo = load_yaml(hpo_path) or {}
    if hpo:
        print(f"Loaded HPO params from: {hpo_path}")
    else:
        print(f"No HPO params found at {hpo_path} — using defaults / CLI overrides")

    # Build train args (minimal, safe)
    train_args = {
        "data": str(data_yaml),
        "weights": weights,
        "epochs": int(args.epochs),
        # 'batch' and 'imgsz' set below (HPO or CLI or defaults)
        "device": args.device,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "save": True,
        "amp": True,        # mixed precision (safe and fast)
        "cos_lr": True,     # recommended by HPO recommended_training.yaml
        "val": True,
        "plots": True,
        "save_period": -1,  # default: don't save periodically beyond best/last
    }

    # Defaults (reasonable)
    defaults = {
        "batch": 16,
        "imgsz": 640,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "mosaic": None,
        "mixup": None,
        "hsv_h": None,
        "hsv_s": None,
        "hsv_v": None,
        "warmup_epochs": None,
    }

    # Apply HPO values (only keys we care about)
    mapping = {
        "lr0": "lr0",
        "lrf": "lrf",
        "batch": "batch",
        "imgsz": "imgsz",
        "optimizer": "optimizer",
        "weight_decay": "weight_decay",
        "box": "box",
        "cls": "cls",
        "dfl": "dfl",
        "mosaic": "mosaic",
        "mixup": "mixup",
        "hsv_h": "hsv_h",
        "hsv_s": "hsv_s",
        "hsv_v": "hsv_v",
        "warmup_epochs": "warmup_epochs",
    }

    # Start from defaults then override with HPO then CLI overrides
    final_params = defaults.copy()
    # HPO
    for k_in, k_out in mapping.items():
        if k_in in hpo and hpo[k_in] is not None:
            final_params[k_out] = hpo[k_in]
    # CLI overrides
    if args.batch_size is not None:
        final_params["batch"] = int(args.batch_size)
    if args.img_size is not None:
        final_params["imgsz"] = int(args.img_size)

    # Put into train_args (only truthy keys)
    for k in ("batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay",
              "box", "cls", "dfl", "mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v",
              "warmup_epochs"):
        v = final_params.get(k, None)
        if v is not None:
            train_args[k] = v

    # Safety clamp: avoid enormous imgsz with big batch
    try:
        if int(train_args.get("batch", 16)) >= 24 and int(train_args.get("imgsz", 640)) > 640:
            print("Large batch with large image size detected — clamping imgsz to 640 for stability.")
            train_args["imgsz"] = 640
    except Exception:
        pass

    # Print summary of what will be used
    print("Final training arguments (minimal):")
    for k in ("data", "weights", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay"):
        if k in train_args:
            print(f"  {k}: {train_args[k]}")
    print("Project directory:", project / name)
    print("Starting training...\n")

    # Initialize model and train
    model = YOLO(train_args.pop("weights"))  # weights moved into YOLO init; train expects 'weights' not in kwargs
    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise

    # Extract metrics robustly
    metrics = _safe_extract_metrics(results)
    score = float(metrics.get("mAP50_95", metrics.get("mAP50", 0.0)))

    # Locate best weights path if possible
    best_weights = find_best_weights_dir(Path(train_args.get("project", project)), name)
    best_weights_str = str(best_weights) if best_weights else "NOT_FOUND"

    # Save summary
    out = {
        "data": str(data_yaml),
        "weights": weights,
        "project": str(project),
        "name": name,
        "hpo_used": bool(hpo),
        "hpo_path": str(hpo_path) if hpo else None,
        "final_train_args": {k: train_args.get(k, final_params.get(k)) for k in train_args.keys()},
        "metrics": metrics,
        "score": score,
        "best_weights": best_weights_str,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = project / name / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"  mAP50:    {metrics.get('mAP50'):.4f}")
    print(f"  mAP50-95: {metrics.get('mAP50_95'):.4f}")
    print(f"  Best weights: {best_weights_str}")
    print(f"  Summary saved to: {summary_path}")
    print("="*60 + "\n")

    return results

if __name__ == "__main__":
    main()
