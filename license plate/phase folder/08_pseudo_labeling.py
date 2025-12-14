#!/usr/bin/env python3
"""
08_pseudo_labeling.py — Senior-Level Version
SAFE RESUME + FAST + NO FP16 FUSION CRASH
"""

import argparse
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import time
import yaml

def ensure_dirs(base: Path):
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)

def safe_open_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return cx/img_w, cy/img_h, w/img_w, h/img_h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", default="datasets/08_pseudo")
    parser.add_argument("--conf-thres", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    # ------------------------------------
    # LOAD TEACHER (NO .half() HERE)
    # ------------------------------------
    teacher = YOLO(args.teacher)
    teacher.to("cuda")  # Let Ultralytics handle autocast FP16 internally

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    ensure_dirs(out)

    # ----------- LOAD PROGRESS -----------
    progress_file = out / "_progress.json"
    start_idx = 0
    if progress_file.exists():
        try:
            start_idx = json.load(open(progress_file))["last_index"] + 1
            print(f"[SAFE RESUME] Resuming from index {start_idx}")
        except:
            start_idx = 0

    # Gather images recursively
    image_paths = sorted([
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    summary = {
        "total_images": len(image_paths),
        "processed": 0,
        "labeled": 0,
        "total_boxes": 0,
        "skipped_small": 0,
        "avg_conf": [],
        "start_time": time.time(),
        "conf_thres": args.conf_thres,
    }

    # ----------- MAIN LOOP -----------
    for idx in range(start_idx, len(image_paths)):
        img_path = image_paths[idx]

        # Skip if already processed
        label_path = out / "labels" / "train" / f"{idx:08d}.txt"
        if label_path.exists():
            continue

        img = safe_open_image(img_path)
        if img is None:
            print(f"[WARN] Corrupted image skipped: {img_path}")
            continue

        w, h = img.size

        # YOLO inference — correct dtype handling
        results = teacher.predict(
            img,
            conf=args.conf_thres,
            imgsz=max(w, h),
            half=True,      # only here
            verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            continue

        det_boxes = []
        conf_list = []

        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])

            if conf < args.conf_thres:
                continue

            bw = x2 - x1
            bh = y2 - y1

            if bw < args.min_size or bh < args.min_size:
                summary["skipped_small"] += 1
                continue

            cx, cy, nw, nh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            det_boxes.append((cls, cx, cy, nw, nh))
            conf_list.append(conf)

        if len(det_boxes) == 0:
            continue

        # Save image
        dst_img = out / "images" / "train" / f"{idx:08d}{img_path.suffix.lower()}"
        if args.copy_mode == "copy":
            shutil.copy2(img_path, dst_img)
        else:
            try:
                dst_img.symlink_to(img_path.resolve())
            except:
                shutil.copy2(img_path, dst_img)

        # Save labels
        with open(label_path, "w") as f:
            for cls, cx, cy, nw, nh in det_boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        summary["processed"] += 1
        summary["labeled"] += 1
        summary["total_boxes"] += len(det_boxes)
        summary["avg_conf"].extend(conf_list)

        # UPDATE SAFE RESUME CHECKPOINT
        if idx % 50 == 0:
            json.dump(
                {"last_index": idx},
                open(progress_file, "w")
            )

    # ----------- FINAL SUMMARY -----------
    summary["end_time"] = time.time()
    summary["runtime_hours"] = (summary["end_time"] - summary["start_time"]) / 3600
    summary["mean_conf"] = float(sum(summary["avg_conf"]) / max(1, len(summary["avg_conf"])))

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write data.yaml
    yaml.dump({
        "path": str(out),
        "train": "images/train",
        "nc": 1,
        "names": ["license_plate"],
    }, open(out / "data.yaml", "w"))

    print("\n✔ Pseudo Labeling Complete!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
