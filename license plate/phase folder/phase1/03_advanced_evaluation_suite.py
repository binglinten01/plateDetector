#!/usr/bin/env python3
"""
03_advanced_evaluation_suite.py

Senior-level evaluation suite for object detector (YOLO11).
Features:
 - Runs standard YOLO validation (mAP)
 - Runs batched inference over val set and:
   * Exports predictions in YOLO txt per image (preds_yolo/)
   * Computes TP / FP / FN by IoU matching
   * Saves examples: FN/, FP/, LC/ (low-conf correct)
   * Produces per-aspect and per-size statistics
 - Speed benchmark: PyTorch FP32, FP16 (and optional ONNX if onnxruntime available)
 - Writes JSON reports, CSV summary, and visualization images
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch

from ultralytics import YOLO

# -------- utilities --------
def xywhn_to_xyxy(center_x, center_y, w, h, img_w, img_h):
    cx = center_x * img_w
    cy = center_y * img_h
    w_px = w * img_w
    h_px = h * img_h
    x1 = float(cx - w_px / 2.0)
    y1 = float(cy - h_px / 2.0)
    x2 = float(cx + w_px / 2.0)
    y2 = float(cy + h_px / 2.0)
    return [x1, y1, x2, y2]

def clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(img_w - 1.0, x1))
    y1 = max(0.0, min(img_h - 1.0, y1))
    x2 = max(0.0, min(img_w - 1.0, x2))
    y2 = max(0.0, min(img_h - 1.0, y2))
    return [x1, y1, x2, y2]

def iou_xyxy(boxA, boxB) -> float:
    # box: [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    areaB = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = areaA + areaB - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

def read_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """
    returns list of [class, x1, y1, x2, y2]
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            xyxy = xywhn_to_xyxy(cx, cy, w, h, img_w, img_h)
            xyxy = clamp_box(xyxy, img_w, img_h)
            boxes.append([cls, *xyxy])
    return boxes

def save_yolo_preds_file(preds_dir: Path, img_name: str, preds: List[Tuple[int, float, List[float]]]):
    """
    preds: list of (cls, conf, [x1,y1,x2,y2]) - xyxy absolute coords
    write normalized YOLO x_center y_center w h format (class conf omitted)
    """
    out = preds_dir / f"{Path(img_name).stem}.txt"
    img = Image.open(img_name)
    w, h = img.size
    with open(out, "w") as f:
        for cls, conf, xyxy in preds:
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {conf:.6f}\n")

def draw_boxes_on_image(img_path: Path, boxes: List[Tuple[str, Tuple[float,float,float,float], float]], out_path: Path):
    """
    boxes: list of (label, (x1,y1,x2,y2), conf)
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for label, (x1,y1,x2,y2), conf in boxes:
        color = (255,0,0) if label.startswith("FP") else (0,255,0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} {conf:.2f}"
        draw.text((x1+4, y1+4), text, fill=color, font=font)
    img.save(out_path)

# -------- main suite --------
def main():
    parser = argparse.ArgumentParser(description="Advanced Evaluation Suite for YOLO detectors (A-E)")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--data", required=True, help="Path to dataset root (contains data.yaml and images/val and labels/val)")
    parser.add_argument("--output", required=True, help="Output directory for evaluation results")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold (very low to capture candidates)")
    parser.add_argument("--lc-conf", type=float, default=0.4, help="Low-confidence threshold for matched correct detections (LC)")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k examples saved for FN/FP")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers for ultralytics predict")
    parser.add_argument("--bench-steps", type=int, default=300, help="Number of images to run in speed benchmark")
    parser.add_argument("--bench-runs", type=int, default=3, help="Repeat bench runs and take median")
    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_root = Path(args.data)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # create structure
    preds_dir = out_root / "preds_yolo"
    preds_dir.mkdir(exist_ok=True)
    examples_dir = out_root / "examples"
    fn_dir = examples_dir / "FN"
    fp_dir = examples_dir / "FP"
    lc_dir = examples_dir / "LC"
    for d in (fn_dir, fp_dir, lc_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    # run standard ultralytics validation (mAP)
    print("\nRunning standard YOLO validation (this may take a while)...")
    try:
        val_metrics = model.val(data=str(dataset_root / "data.yaml"))
        # ultralytics returns object that may have metrics in .box
        std_metrics = {
            "mAP50": float(getattr(getattr(val_metrics, "box", val_metrics), "map50", getattr(val_metrics, "map50", 0.0))),
            "mAP50_95": float(getattr(getattr(val_metrics, "box", val_metrics), "map", getattr(val_metrics, "map", 0.0))),
            "precision": float(getattr(getattr(val_metrics, "box", val_metrics), "p", getattr(val_metrics, "p", 0.0))),
            "recall": float(getattr(getattr(val_metrics, "box", val_metrics), "r", getattr(val_metrics, "r", 0.0))),
        }
    except Exception as e:
        print(f"Standard validation failed: {e}")
        std_metrics = {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None}

    # collect image list for val
    val_images_dir = dataset_root / "images" / "val"
    val_labels_dir = dataset_root / "labels" / "val"
    if not val_images_dir.exists() or not val_labels_dir.exists():
        print(f"ERROR: expected dataset layout: images/val and labels/val under {dataset_root}")
        sys.exit(1)
    image_paths = sorted([p for p in val_images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    n_images = len(image_paths)
    print(f"\nVal images found: {n_images}")

    # inference (batched) using model.predict - collect results
    print("\nRunning inference over val set (collecting preds)...")
    # Use ultralytics predict in batched mode for speed: provide source=dir and conf=...
    # But we need per-image preds objects; model.predict returns list of Results same order as files
    res_list = model.predict(source=str(val_images_dir), conf=args.conf, save=False, verbose=False, imgsz=640, stream=False)
    # model.predict may return [] if something odd; ensure len matches
    if not res_list or len(res_list) != n_images:
        # fallback: run per-image (slower) but robust
        print("Fallback: per-image inference (slower)")
        res_list = []
        for p in image_paths:
            res = model.predict(source=str(p), conf=args.conf, save=False, verbose=False)
            res_list.append(res[0] if res else None)

    # prepare accumulators
    all_stats = {
        "total_images": n_images,
        "total_gt": 0,
        "total_preds": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
    }
    per_image_records = []

    # for per-size and per-aspect analysis
    aspect_buckets = {"square": [], "landscape": [], "wide": [], "portrait": []}
    size_buckets = {"tiny": [], "small": [], "medium": [], "large": []}

    # iterate images and compute matching
    print("\nMatching predictions to ground-truth (IoU threshold = %.2f) ..." % args.iou)
    for idx, img_path in enumerate(image_paths):
        img_name = img_path.name
        img = Image.open(img_path)
        img_w, img_h = img.size

        # read GT
        label_file = val_labels_dir / f"{Path(img_name).stem}.txt"
        gt_boxes = read_yolo_label_file(label_file, img_w, img_h)  # list [class,x1,y1,x2,y2]
        all_stats["total_gt"] += len(gt_boxes)

        # predictions
        res = res_list[idx]
        preds = []
        if res is None:
            preds = []
        else:
            # res.boxes: list of Box objects
            for b in getattr(res, "boxes", []):
                # some builds b.xyxy may be tensor; handle robustly
                try:
                    xy = b.xyxy[0].cpu().numpy().tolist()
                except Exception:
                    try:
                        xy = b.xyxy.tolist()[0]
                    except Exception:
                        xy = [float(v) for v in b.xyxy[0]]
                conf = float(getattr(b, "conf", 1.0)[0]) if getattr(b, "conf", None) is not None else float(getattr(b, "confidence", 1.0))
                cls = int(getattr(b, "cls", 0)[0]) if getattr(b, "cls", None) is not None else 0
                xy = clamp_box(xy, img_w, img_h)
                preds.append((cls, conf, xy))

        all_stats["total_preds"] += len(preds)
        # export YOLO-style preds (with conf)
        save_yolo_preds_file(preds_dir, str(img_path), preds)

        # create IoU matrix (gt x preds)
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(preds)
        iou_mat = np.zeros((len(gt_boxes), len(preds)), dtype=float)

        for i, gt in enumerate(gt_boxes):
            _, gx1, gy1, gx2, gy2 = gt
            for j, pred in enumerate(preds):
                _, pconf, (px1, py1, px2, py2) = pred
                iou_mat[i, j] = iou_xyxy([gx1, gy1, gx2, gy2], [px1, py1, px2, py2])

        # greedy matching: match highest IoU first
        matches = []
        if iou_mat.size > 0:
            # flatten sorted by IoU descending
            idxs = np.unravel_index(np.argsort(iou_mat.ravel())[::-1], iou_mat.shape)
            for gi, pj in zip(idxs[0], idxs[1]):
                if iou_mat[gi, pj] < args.iou:
                    break
                if not gt_matched[gi] and not pred_matched[pj]:
                    gt_matched[gi] = True
                    pred_matched[pj] = True
                    matches.append((gi, pj, float(iou_mat[gi, pj])))

        # count stats
        tp = sum(pred_matched)
        fp = len(preds) - tp
        fn = len(gt_boxes) - sum(gt_matched)
        all_stats["TP"] += tp
        all_stats["FP"] += fp
        all_stats["FN"] += fn

        # analyze LC: low-conf matched preds
        lc_examples = []
        fp_examples = []
        fn_examples = []

        # create records for each matched pair
        for (gi, pj, iouv) in matches:
            cls_gt, gx1, gy1, gx2, gy2 = gt_boxes[gi]
            cls_p, pconf, pxy = preds[pj]
            # low-confidence correct
            if pconf < args.lc_conf:
                lc_examples.append((pj, pconf, pxy, (gx1, gy1, gx2, gy2), iouv))
        # false positives: unmatched preds
        for j, pred in enumerate(preds):
            if not pred_matched[j]:
                cls_p, pconf, pxy = pred
                # choose FP examples by high confidence
                fp_examples.append((j, pconf, pxy))
        # false negatives: unmatched GTs
        for i, g in enumerate(gt_boxes):
            if not gt_matched[i]:
                cls_gt, gx1, gy1, gx2, gy2 = g
                area = (gx2 - gx1) * (gy2 - gy1)
                fn_examples.append((i, area, (gx1, gy1, gx2, gy2)))

        # save top-k per image examples into per-image records (we'll aggregate top overall later)
        per_image_records.append({
            "img": str(img_path),
            "n_gt": len(gt_boxes),
            "n_pred": len(preds),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "matches": matches,
            "lc": lc_examples,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples
        })

        # accumulate analysis buckets from gt boxes (aspect, size)
        for g in gt_boxes:
            _, gx1, gy1, gx2, gy2 = g
            w = max(1e-6, gx2 - gx1)
            h = max(1e-6, gy2 - gy1)
            ar = w / h
            area_rel = (w * h) / (img_w * img_h)
            # aspect
            if 0.8 <= ar <= 1.2:
                aspect_buckets["square"].append(ar)
            elif 1.2 < ar <= 2.0:
                aspect_buckets["landscape"].append(ar)
            elif ar > 2.0:
                aspect_buckets["wide"].append(ar)
            else:
                aspect_buckets["portrait"].append(ar)
            # size
            if area_rel < 0.01:
                size_buckets["tiny"].append(area_rel)
            elif area_rel < 0.05:
                size_buckets["small"].append(area_rel)
            elif area_rel < 0.1:
                size_buckets["medium"].append(area_rel)
            else:
                size_buckets["large"].append(area_rel)

    # done loop
    print("\nMatching complete. Aggregating statistics...")

    # aggregate and save per-image records
    with open(metrics_dir / "per_image_records.json", "w") as f:
        def to_py(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: to_py(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_py(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(to_py(v) for v in obj)
            return obj

        json.dump(to_py(per_image_records), f, indent=2)

    # create global report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(model_path),
        "dataset": str(dataset_root),
        "standard_metrics": std_metrics,
        "aggregate": all_stats,
        "iou_threshold": args.iou,
        "conf_threshold": args.conf,
        "low_conf_threshold": args.lc_conf,
        "aspect_buckets": {k: len(v) for k, v in aspect_buckets.items()},
        "size_buckets": {k: len(v) for k, v in size_buckets.items()},
    }
    with open(metrics_dir / "aggregate_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # pick top-K FPs (by confidence) and FNs (by area)
    print("\nExporting top examples (FP / FN / LC)...")
    # gather all FP candidates / FN candidates / LC candidates
    fp_candidates = []
    fn_candidates = []
    lc_candidates = []
    for rec in per_image_records:
        img = Path(rec["img"])
        for idx, pconf, pxy in rec["fp_examples"]:
            fp_candidates.append((rec["img"], float(pconf), pxy))
        for idx, area, gxy in rec["fn_examples"]:
            fn_candidates.append((rec["img"], float(area), gxy))
        for idx, pconf, pxy, gxy, iouv in rec["lc"]:
            lc_candidates.append((rec["img"], float(pconf), pxy, gxy, float(iouv)))

    # sort
    fp_candidates = sorted(fp_candidates, key=lambda x: x[1], reverse=True)[:args.top_k]
    fn_candidates = sorted(fn_candidates, key=lambda x: x[1], reverse=True)[:args.top_k]  # large misses first
    lc_candidates = sorted(lc_candidates, key=lambda x: x[1])[:args.top_k]  # lowest conf first

    # save examples to folders with drawn boxes
    for i, (img_path, pconf, pxy) in enumerate(fp_candidates):
        src = Path(img_path)
        dst = fp_dir / f"{i:04d}_{src.name}"
        # draw pred box as FP
        label = f"FP {pconf:.3f}"
        draw_boxes_on_image(src, [(label, tuple(pxy), pconf)], dst)
    for i, (img_path, area, gxy) in enumerate(fn_candidates):
        src = Path(img_path)
        dst = fn_dir / f"{i:04d}_{src.name}"
        label = f"FN area={area:.0f}"
        draw_boxes_on_image(src, [(label, tuple(gxy), 0.0)], dst)
    for i, (img_path, pconf, pxy, gxy, iouv) in enumerate(lc_candidates):
        src = Path(img_path)
        dst = lc_dir / f"{i:04d}_{src.name}"
        boxes = [("GT", tuple(gxy), 1.0), (f"LC_pred {pconf:.3f}", tuple(pxy), pconf)]
        draw_boxes_on_image(src, boxes, dst)

    # save preds archive (optionally zip?) - we keep files in preds_yolo
    # Save summary
    with open(out_root / "evaluation_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # -------- Speed benchmark ----------
    bench = {"pytorch_fp32": {}, "pytorch_fp16": {}}
    print("\n🏁 Running speed benchmarks (PyTorch FP32 & FP16)...")
    imgs_for_bench = image_paths[: max(1, min(len(image_paths), args.bench_steps))]
    # helper to run inference loop and measure fps
    def bench_run(model_obj, imgs, half=False):
        # model_obj: ultralytics YOLO instance
        # run warmup
        for _ in range(5):
            model_obj.predict(source=str(imgs[0]), imgsz=640, conf=args.conf, save=False, verbose=False)
        niters = len(imgs)
        t0 = time.time()
        for p in imgs:
            model_obj.predict(source=str(p), imgsz=640, conf=args.conf, save=False, verbose=False)
        t1 = time.time()
        elapsed = t1 - t0
        fps = niters / elapsed if elapsed > 0 else 0.0
        return {"elapsed_s": elapsed, "fps": fps, "n": niters}

    # FP32
    try:
        model_cpu = YOLO(str(model_path))  # fresh model (will auto select device)
        # ensure model on GPU if available
        if torch.cuda.is_available():
            model_cpu.to("cuda")
        # ensure float32
        # run bench multiple times
        fps_list = []
        for _ in range(args.bench_runs):
            r = bench_run(model_cpu, imgs_for_bench, half=False)
            fps_list.append(r["fps"])
        bench["pytorch_fp32"]["fps_median"] = float(np.median(fps_list))
        bench["pytorch_fp32"]["runs"] = fps_list
    except Exception as e:
        bench["pytorch_fp32"]["error"] = str(e)

    # FP16 (half) - only if CUDA available
    if torch.cuda.is_available():
        try:
            model_half = YOLO(str(model_path))
            model_half.to("cuda")
            # set half where possible by passing half=True in predict calls; ultralytics may also accept model_half.model.half()
            try:
                model_half.model.half()
            except Exception:
                pass
            fps_list = []
            for _ in range(args.bench_runs):
                r = bench_run(model_half, imgs_for_bench, half=True)
                fps_list.append(r["fps"])
            bench["pytorch_fp16"]["fps_median"] = float(np.median(fps_list))
            bench["pytorch_fp16"]["runs"] = fps_list
        except Exception as e:
            bench["pytorch_fp16"]["error"] = str(e)
    else:
        bench["pytorch_fp16"]["error"] = "cuda_not_available"

    # ONNX benchmark (optional) - try if onnxruntime installed
    try:
        import onnxruntime as ort
        # attempt to export ONNX to temp file
        onnx_path = out_root / "tmp_model.onnx"
        if not onnx_path.exists():
            try:
                model.export(format="onnx", imgsz=640, verbose=False, half=False, save=True)
                # ultralytics export writes to runs/export; try to find newest onnx
                # fallback: we won't rely on strict path; attempt to locate in run directories
                # This step is best-effort
            except Exception:
                pass
        # search for an onnx file in project
        candidate = None
        for p in (Path.cwd().rglob("*.onnx")):
            # pick the newest small onnx produced closely
            candidate = p
            break
        if candidate:
            sess = ort.InferenceSession(str(candidate))
            # prepare single image input
            import numpy as _np
            p0 = imgs_for_bench[0]
            img = cv2.imread(str(p0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Preprocess: simple resizing to 640 keep aspect? We'll do center-crop/resize to 640x640
            img_resized = cv2.resize(img, (640, 640))
            inp = _np.transpose(img_resized, (2, 0, 1)).astype("float32") / 255.0
            inp = inp[np.newaxis, :]
            # run ONNX bench
            fps_list = []
            for _ in range(args.bench_runs):
                t0 = time.time()
                for _i in range(len(imgs_for_bench)):
                    sess.run(None, {sess.get_inputs()[0].name: inp})
                t1 = time.time()
                fps_list.append(len(imgs_for_bench) / (t1 - t0))
            bench["onnx"] = {"fps_median": float(np.median(fps_list)), "runs": fps_list, "onnx_path": str(candidate)}
        else:
            bench["onnx"] = {"error": "onnx_not_found_after_export_attempt"}
    except Exception as e:
        bench["onnx"] = {"error": f"onnxruntime_unavailable_or_failed: {e}"}

    # save bench
    with open(metrics_dir / "bench.json", "w") as f:
        json.dump(bench, f, indent=2)

    # finalize full report
    final_report = {
        "report": report,
        "bench": bench,
    }
    with open(out_root / "full_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n✅ Advanced evaluation complete!")
    print(f"Report & artifacts saved under: {out_root}")

if __name__ == "__main__":
    main()
