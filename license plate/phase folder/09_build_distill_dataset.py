#!/usr/bin/env python3
"""
09_build_distill_dataset.py

Senior-level dataset builder for distillation training.

What it does:
 - Creates `09_distill_training/` (or user-specified outdir)
 - Merges clean teacher dataset (07_final_training) + pseudo dataset (08_pseudo)
 - Keeps teacher's val/test splits (if present) and appends pseudo only to train
 - Removes duplicate images (by MD5) preferring teacher labels over pseudo
 - Normalizes filenames (00000001.jpg, corresponding .txt labels)
 - Supports copy or symlink mode
 - Validates YOLO label files (basic format check)
 - Produces `data.yaml` ready for Ultralytics training
 - Writes a summary JSON and a mapping CSV for traceability

Usage example:
    python 09_build_distill_dataset.py \
      --clean datasets/07_final_training \
      --pseudo datasets/08_pseudo \
      --out datasets/09_distill_training \
      --copy-mode symlink

Notes:
 - Assumes dataset structure like: <dataset>/images/(train|val|test)/ and <dataset>/labels/(train|val|test)/
 - Pseudo dataset may only contain train/labels. We only add pseudo files into the train split.
 - Teacher (clean) val/test are preserved and won't be mixed with pseudo.

"""

from pathlib import Path
import argparse
import hashlib
import shutil
import json
import csv
import sys
from typing import Dict, Tuple

CHUNK_SIZE = 8192
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_split_dirs(root: Path) -> Dict[str, Tuple[Path, Path]]:
    """Return mapping split -> (images_dir, labels_dir) for train/val/test if exist."""
    out = {}
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if img_dir.exists() and any(img_dir.rglob("*")):
            out[split] = (img_dir, lbl_dir)
    return out


def ensure_out_dirs(out: Path):
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def validate_label_file(p: Path) -> bool:
    """Basic validation: each line has at least 5 numeric tokens (class and 4 floats)
    Returns True if file seems valid.
    """
    try:
        text = p.read_text(encoding="utf-8").strip()
        if text == "":
            return False
        for i, line in enumerate(text.splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            # class int and rest floats
            int(parts[0])
            for tok in parts[1:5]:
                float(tok)
        return True
    except Exception:
        return False


def copy_or_symlink(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            # fallback to copy
            shutil.copy2(src, dst)


def gather_images_labels(img_dir: Path, lbl_dir: Path):
    """Yield (img_path, label_path)
    If label doesn't exist, label_path will be None.
    """
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in VALID_IMAGE_EXT:
            continue
        # name without ext
        stem = img.stem
        lbl = None
        # check for .txt
        txt = lbl_dir / f"{stem}.txt"
        if txt.exists():
            lbl = txt
        yield img, lbl


def main():
    parser = argparse.ArgumentParser(description="Build distillation dataset by merging clean + pseudo")
    parser.add_argument("--clean", required=True, help="Clean teacher dataset root (07_final_training)")
    parser.add_argument("--pseudo", required=True, help="Pseudo dataset root (08_pseudo)")
    parser.add_argument("--out", default="datasets/09_distill_training", help="Output dataset root")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink", help="Whether to copy images or symlink")
    parser.add_argument("--keep-empty-labels", action="store_true", help="Keep images that have no labels (not recommended)")
    parser.add_argument("--dedupe-by", choices=["md5", "filename"], default="md5", help="How to detect duplicates (default md5)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for normalized filenames (default 0)")
    args = parser.parse_args()

    clean_root = Path(args.clean)
    pseudo_root = Path(args.pseudo)
    out_root = Path(args.out)
    ensure_out_dirs(out_root)

    # gather splits from clean
    clean_splits = find_split_dirs(clean_root)
    pseudo_splits = find_split_dirs(pseudo_root)

    # We'll preserve clean val/test if available
    preserved_splits = [s for s in ("val", "test") if s in clean_splits]

    # track seen images by md5 or filename
    seen = {}
    mapping_rows = []
    summary = {"clean_added": 0, "pseudo_added": 0, "duplicates_skipped": 0, "invalid_labels": 0}

    next_idx = args.start_index

    def make_name(idx: int, ext: str) -> str:
        return f"{idx:08d}{ext.lower()}"

    # 1) Add clean splits (train/val/test) — prefer these if duplicates occur
    for split, paths in clean_splits.items():
        img_dir, lbl_dir = paths
        for img, lbl in gather_images_labels(img_dir, lbl_dir):
            key = img.name if args.dedupe_by == "filename" else md5_of_file(img)
            if key in seen:
                summary["duplicates_skipped"] += 1
                continue
            # validate label exists
            if lbl is None:
                if not args.keep_empty_labels:
                    summary["invalid_labels"] += 1
                    continue
            else:
                if not validate_label_file(lbl):
                    summary["invalid_labels"] += 1
                    continue

            new_img_name = make_name(next_idx, img.suffix)
            new_lbl_name = make_name(next_idx, ".txt")
            dst_img = out_root / "images" / split / new_img_name
            dst_lbl = out_root / "labels" / split / new_lbl_name

            copy_or_symlink(img, dst_img, args.copy_mode)
            if lbl is not None:
                copy_or_symlink(lbl, dst_lbl, args.copy_mode)
            else:
                # create empty file only if keep_empty
                if args.keep_empty_labels:
                    dst_lbl.write_text("")

            seen[key] = {"source": "clean", "orig_image": str(img), "orig_label": str(lbl) if lbl else None}
            mapping_rows.append((new_img_name, "clean", str(img), str(lbl) if lbl else ""))
            next_idx += 1
            summary["clean_added"] += 1

    # 2) Add pseudo to train split only
    if "train" in pseudo_splits:
        img_dir, lbl_dir = pseudo_splits["train"]
        for img, lbl in gather_images_labels(img_dir, lbl_dir):
            key = img.name if args.dedupe_by == "filename" else md5_of_file(img)
            if key in seen:
                # prefer clean if already present
                summary["duplicates_skipped"] += 1
                continue
            # require label for pseudo
            if lbl is None:
                summary["invalid_labels"] += 1
                continue
            if not validate_label_file(lbl):
                summary["invalid_labels"] += 1
                continue

            split = "train"
            new_img_name = make_name(next_idx, img.suffix)
            new_lbl_name = make_name(next_idx, ".txt")
            dst_img = out_root / "images" / split / new_img_name
            dst_lbl = out_root / "labels" / split / new_lbl_name

            copy_or_symlink(img, dst_img, args.copy_mode)
            copy_or_symlink(lbl, dst_lbl, args.copy_mode)

            seen[key] = {"source": "pseudo", "orig_image": str(img), "orig_label": str(lbl)}
            mapping_rows.append((new_img_name, "pseudo", str(img), str(lbl)))
            next_idx += 1
            summary["pseudo_added"] += 1

    # 3) If clean had no val/test, optionally create a small val from combined set (not implemented automatically)

    # 4) Write data.yaml using preserved splits (prefer clean val/test). If none, use train only.
    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "nc": 1,
        "names": ["license_plate"]
    }
    if "val" in clean_splits:
        data_yaml["val"] = "images/val"
    if "test" in clean_splits:
        data_yaml["test"] = "images/test"

    # write files
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        json.dump(data_yaml, f, indent=2)

    # mapping csv
    with open(out_root / "mapping.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_filename", "source", "orig_image", "orig_label"])
        for row in mapping_rows:
            writer.writerow(row)

    # summary file
    summary["total_final_images"] = next_idx - args.start_index
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n✔ Distillation dataset built:")
    print(f"  Out: {out_root}")
    print(f"  Clean added: {summary['clean_added']}")
    print(f"  Pseudo added: {summary['pseudo_added']}")
    print(f"  Duplicates skipped: {summary['duplicates_skipped']}")
    print(f"  Invalid labels skipped: {summary['invalid_labels']}")
    print(f"  Final images: {summary['total_final_images']}")
    print("  Mapping: mapping.csv, summary.json")


if __name__ == '__main__':
    main()
