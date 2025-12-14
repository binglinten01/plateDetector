#!/usr/bin/env python3
# scripts/phase1/02_knowledge_distillation_full.py

import argparse
import json
from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Knowledge Distillation: Train student on merged clean+pseudo dataset")

    p.add_argument("--teacher", type=str, default=None)
    p.add_argument("--student", type=str, default="yolo11n.pt")
    p.add_argument("--data", type=str, required=True)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img", type=int, default=640)

    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lrf", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="auto")

    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/distill_students")
    p.add_argument("--name", type=str, default="student_distilled")
    p.add_argument("--exist_ok", action="store_true")

    p.add_argument("--resume", action="store_true", help="Resume training")
    p.add_argument("--save-period", type=int, default=-1)

    return p.parse_args()


def save_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()

    data_yaml = Path(args.data)
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"

    print("\n======================================================================")
    print(" STARTING STUDENT DISTILLATION (supervised)")
    print(f" Teacher: {args.teacher}")
    print(f" Student init: {args.student}")
    print(f" Dataset: {args.data}")
    print(f" Run: {args.project}/{args.name}")
    print("======================================================================\n")

    # Load student model
    student = YOLO(args.student)

    # Valid arguments for YOLO().train():
    train_args = {
        "model": args.student,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.img,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,

        "optimizer": args.optimizer,
        "lr0": args.lr,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,

        "resume": args.resume,
        "cos_lr": True,
        "patience": 100,
        "pretrained": True,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": args.save_period,
    }

    # Train
    results = student.train(**train_args)

    # Extract metrics
    metrics = {}
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        for k in ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"]:
            if k in rd:
                metrics[k] = rd[k]

    run_dir = Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    best_path = str(best) if best.exists() else (str(last) if last.exists() else None)

    summary = {
        "teacher": args.teacher,
        "student_init": args.student,
        "data": str(data_yaml),
        "project": args.project,
        "name": args.name,
        "train_args": train_args,
        "metrics": metrics,
        "best_weights": best_path,
        "timestamp": datetime.now().isoformat()
    }

    save_summary(run_dir / "training_summary.json", summary)

    print("\n======================================================================")
    print(" DISTILLATION COMPLETE")
    print(f" Best weights: {best_path}")
    print(f" Summary saved to: {run_dir / 'training_summary.json'}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
