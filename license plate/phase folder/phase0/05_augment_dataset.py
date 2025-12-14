# scripts/05_augment_dataset_final.py
"""
ULTIMATE AUGMENTATION PIPELINE - FIXED ALBUMENTATIONS API
Updated for latest albumentations version
"""

import argparse
import json
import logging
import random
import shutil
import warnings  # THÊM DÒNG NÀY
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import yaml
import albumentations as A
from tqdm import tqdm

# ---------------------------
# Logger
# ---------------------------
def setup_logger(name: str = "augmenter", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

logger = setup_logger()


# ---------------------------
# Utility functions (ENHANCED)
# ---------------------------
def read_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def load_yolo_label(path: Path) -> List[List[float]]:
    """Return list of lines: [class, x_center, y_center, w, h] as floats"""
    boxes = []
    if not path.exists():
        return boxes
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    vals = [float(x) for x in parts[:5]]
                    boxes.append(vals)
    except Exception as e:
        logger.warning(f"Failed to read label {path}: {e}")
    return boxes

def save_yolo_label(path: Path, boxes: List[List[float]]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            for b in boxes:
                # b: [class, x, y, w, h]
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
    except Exception as e:
        logger.error(f"Failed to write label {path}: {e}")

def calculate_blur_score(image: np.ndarray) -> float:
    """Enhanced blur calculation"""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_brightness_contrast(image: np.ndarray) -> Tuple[float, float]:
    """Calculate brightness and contrast"""
    if image is None or image.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return float(brightness), float(contrast)

def normalized_boxes_ok(boxes: List[List[float]]) -> bool:
    """Check if boxes are properly normalized"""
    for b in boxes:
        _, x, y, w, h = b
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False
        # Additional check: plate aspect ratio typically 2:1 to 4:1
        if h > 0:
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                logger.debug(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    return True

def validate_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Comprehensive image quality validation"""
    quality = {
        'blur_score': calculate_blur_score(image),
        'brightness': 0.0,
        'contrast': 0.0,
        'is_valid': True
    }
    
    brightness, contrast = calculate_brightness_contrast(image)
    quality['brightness'] = brightness
    quality['contrast'] = contrast
    
    # Quality thresholds
    if quality['blur_score'] < 50:
        quality['is_valid'] = False
        logger.debug(f"Low blur score: {quality['blur_score']:.1f}")
    
    if brightness < 30 or brightness > 220:
        quality['is_valid'] = False
        logger.debug(f"Extreme brightness: {brightness:.1f}")
    
    if contrast < 20:
        quality['is_valid'] = False
        logger.debug(f"Low contrast: {contrast:.1f}")
    
    return quality


# ---------------------------
# Helper function để test API (ĐÃ SỬA)
# ---------------------------
def test_albu_api():
    """Test Albumentations 2.0.8 API compatibility"""
    import albumentations as A
    
    print("🧪 Testing Albumentations 2.0.8 API...")
    
    # Test GaussNoise
    try:
        gn = A.GaussNoise(var_limit=(10.0, 20.0))
        print("✅ GaussNoise: var_limit=(min, max) works")
    except Exception as e:
        print(f"❌ GaussNoise failed: {e}")
        # Thử alternative
        try:
            gn = A.GaussNoise(var_limit=10.0)
            print("✅ GaussNoise: var_limit=single_value works")
        except Exception as e2:
            print(f"❌ GaussNoise alternative failed: {e2}")
    
    # Test RandomShadow
    try:
        rs = A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows=(1, 2))
        print("✅ RandomShadow: num_shadows=(lower, upper) works")
    except Exception as e:
        print(f"❌ RandomShadow failed: {e}")
        # Thử old API
        try:
            rs = A.RandomShadow(shadow_roi=(0, 0.5, 1, 1))
            print("✅ RandomShadow: default parameters work")
        except Exception as e2:
            print(f"❌ RandomShadow alternative failed: {e2}")
    
    print(f"📦 Albumentations version: {A.__version__}")


# ---------------------------
# Augmenter class (FIXED ALBUMENTATIONS API)
# ---------------------------
class UltimateDatasetAugmenter:
    def __init__(self, config_path: str, multiplier: float = 2.0, seed: int = 42):
        self.config = read_yaml(Path(config_path))
        self.cleaned_dir = Path(self.config["paths"]["cleaned_data"])
        self.augmented_dir = Path(self.config["paths"].get("augmented_data", "datasets/04_augmented"))
        self.multiplier = multiplier
        self.seed = int(seed)
        
        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Check if cleaned dataset exists
        if not self.cleaned_dir.exists():
            logger.error(f"❌ Cleaned dataset not found: {self.cleaned_dir}")
            logger.error("Please run cleaning pipeline first: python scripts/04_clean_dataset.py")
            return
        
        # Initialize
        self._init_pipelines()
        ensure_dir(self.augmented_dir / "images")
        ensure_dir(self.augmented_dir / "labels")

        # Enhanced report
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "multiplier": self.multiplier,
            "splits": {},
            "pipeline_usage": defaultdict(int),
            "quality_metrics": defaultdict(list),
            "failed_augmentations": 0,
            "invalid_images_skipped": 0,
            "albu_version": A.__version__  # Track version
        }

        logger.info(f"🚀 Ultimate Augmenter v2.0 (seed={self.seed})")
        logger.info(f"📁 Cleaned dir: {self.cleaned_dir}")
        logger.info(f"📁 Augmented dir: {self.augmented_dir}")
        logger.info(f"📦 Albumentations version: {A.__version__}")

    def _init_pipelines(self):
        """Initialize augmentation pipelines - SIMPLIFIED for Albumentations 2.0.8"""
        bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.25)
        
        logger.info(f"📊 Albumentations version: {A.__version__}")
        
        # Pipeline weights (more weight to conservative pipelines)
        self.pipeline_names = ["mild_color", "mild_geo", "blur_noise", "weather_light", 
                            "combo_light", "advanced", "rare_cases_light"]
        self.pipeline_weights = [0.18, 0.18, 0.15, 0.12, 0.15, 0.15, 0.07]  # sum=1
        
        # SIMPLIFIED VERSION: Sử dụng API an toàn, không cảnh báo
        # TẠO BIẾN ĐƠN GIẢN TRƯỚC
        gauss_noise_params = {"p": 0.3}  # Mặc định
        shadow_params = {"shadow_roi": (0, 0.5, 1, 1), "p": 0.25}  # Mặc định
        
        # Tạo pipelines
        self.pipelines = {
            "mild_color": A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.2),
            ], bbox_params=bbox_params),

            "mild_geo": A.Compose([
                A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), 
                        rotate=(-4, 4), shear=(-3,3), p=0.6, fit_output=True),
                A.Perspective(scale=(0.01, 0.03), keep_size=True, p=0.25),
            ], bbox_params=bbox_params),

            "blur_noise": A.Compose([
                A.GaussianBlur(blur_limit=(3,5), p=0.35),
                A.MotionBlur(blur_limit=3, p=0.15),
                A.GaussNoise(p=0.3)  # Mặc định
            ], bbox_params=bbox_params),

            "weather_light": A.Compose([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.25),  # Mặc định
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ], bbox_params=bbox_params),

            "combo_light": A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0)
                ], p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.6),
                A.Affine(scale=(0.97, 1.03), translate_percent=(-0.02, 0.02), 
                        rotate=(-3, 3), p=0.5)
            ], bbox_params=bbox_params),

            "advanced": A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),  # Mặc định
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.5),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, 
                                    val_shift_limit=15, p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), 
                        rotate=(-8, 8), shear=(-6, 6), p=0.5)
            ], bbox_params=bbox_params),

            "rare_cases_light": A.Compose([
                A.Affine(scale=(0.85, 0.95), rotate=(-15, 15), 
                        shear=(-10, 10), p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.GaussNoise(p=0.5),  # Mặc định
            ], bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.2
            ))
        }
        
        logger.info("✅ Pipelines initialized with simplified API")

    def _select_pipeline(self):
        """Select pipeline with weighted random choice"""
        return random.choices(self.pipeline_names, weights=self.pipeline_weights, k=1)[0]

    def _denormalize_if_needed(self, raw_boxes: List[List[float]], image_w: int, image_h: int):
        """Auto-detect and convert non-normalized boxes"""
        normalized_boxes = []
        
        for box in raw_boxes:
            cls, x, y, w, h = box
            
            # Check if coordinates need normalization
            if w > 1.0 or h > 1.0:
                # Assume absolute coordinates, convert to normalized
                nx = x / image_w
                ny = y / image_h
                nw = w / image_w
                nh = h / image_h
                normalized_boxes.append([int(cls), nx, ny, nw, nh])
                logger.debug(f"Denormalized box: {w}x{h} -> {nw:.3f}x{nh:.3f}")
            else:
                # Already normalized
                normalized_boxes.append([int(cls), float(x), float(y), float(w), float(h)])
        
        return normalized_boxes

    def augment_all_splits(self):
        """Main augmentation pipeline"""
        splits = ["train", "val", "test"]
        
        for split in splits:
            self.augment_split(split)

        # Finalize report
        self.report["pipeline_usage"] = dict(self.report["pipeline_usage"])
        
        # Calculate quality statistics
        self._calculate_quality_statistics()
        
        # Save reports
        report_path = self.augmented_dir / "ultimate_augmentation_report_v2.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Augmentation report saved to: {report_path}")
        
        # Create data.yaml
        self._write_data_yaml()
        
        # Print summary
        self._print_summary()

    def augment_split(self, split: str):
        """Augment a single split with enhanced safety"""
        logger.info(f"🔧 Processing split: {split}")
        
        src_images = self.cleaned_dir / "images" / split
        src_labels = self.cleaned_dir / "labels" / split
        dst_images = self.augmented_dir / "images" / split
        dst_labels = self.augmented_dir / "labels" / split

        ensure_dir(dst_images)
        ensure_dir(dst_labels)

        if not src_images.exists() or not src_labels.exists():
            logger.warning(f"⚠️ Split missing: {split}")
            self.report["splits"][split] = {"original_count": 0, "augmented_count": 0}
            return

        # Get all images
        image_files = sorted([p for p in src_images.iterdir() if is_image_file(p)])
        original_count = len(image_files)
        
        if original_count == 0:
            logger.warning(f"⚠️ No images found for split: {split}")
            self.report["splits"][split] = {"original_count": 0, "augmented_count": 0}
            return

        # Calculate augmentation needs
        target_count = int(original_count * self.multiplier)
        augmentations_needed = max(0, target_count - original_count)
        
        logger.info(f"📈 Target: {original_count} → {target_count} images (+{augmentations_needed})")

        # 1. COPY ORIGINALS (with safety checks)
        copied_originals = 0
        skipped_originals = 0
        
        pbar_copy = tqdm(image_files, desc=f"Copying originals {split}")
        for img_path in pbar_copy:
            lbl_path = src_labels / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                skipped_originals += 1
                continue
            
            # Load and validate original image
            image = cv2.imread(str(img_path))
            if image is None:
                skipped_originals += 1
                continue
            
            # Validate image quality
            quality = validate_image_quality(image)
            if not quality['is_valid']:
                self.report["invalid_images_skipped"] += 1
                skipped_originals += 1
                continue
            
            # Copy valid original
            shutil.copy2(img_path, dst_images / img_path.name)
            shutil.copy2(lbl_path, dst_labels / lbl_path.name)
            copied_originals += 1
            
            # Update progress
            pbar_copy.set_postfix({"copied": copied_originals, "skipped": skipped_originals})
        
        pbar_copy.close()
        
        logger.info(f"📥 Copied {copied_originals} originals ({skipped_originals} skipped)")

        # 2. PERFORM AUGMENTATIONS (with enhanced safety)
        if augmentations_needed == 0:
            logger.info(f"✅ No augmentation needed for {split}")
            self.report["splits"][split] = {
                "original_count": original_count,
                "skipped_originals": skipped_originals,
                "copied_originals": copied_originals,
                "augmented_count": 0,
                "final_total": copied_originals,
                "augmentation_ratio": 0
            }
            return
        
        augmented_count = 0
        idx = 0
        failed_attempts = 0
        max_failed_attempts = augmentations_needed * 3  # Allow some retries
        
        pbar = tqdm(total=augmentations_needed, desc=f"Augmenting {split}", unit="img")
        
        while augmented_count < augmentations_needed and failed_attempts < max_failed_attempts:
            if idx >= len(image_files):
                idx = 0  # cycle
            
            img_path = image_files[idx]
            idx += 1
            
            lbl_path = src_labels / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Load and validate labels
            raw_boxes = load_yolo_label(lbl_path)
            if not raw_boxes:
                continue
            
            # Auto-detect and normalize boxes
            normalized_boxes = self._denormalize_if_needed(raw_boxes, w, h)
            
            # Validate normalized boxes
            if not normalized_boxes_ok(normalized_boxes):
                failed_attempts += 1
                continue
            
            # Select and apply pipeline
            pipeline_name = self._select_pipeline()
            pipeline = self.pipelines[pipeline_name]
            
            try:
                transformed = pipeline(
                    image=image,
                    bboxes=[b[1:] for b in normalized_boxes],
                    class_labels=[b[0] for b in normalized_boxes]
                )
            except Exception as e:
                logger.debug(f"⚠️ Augmentation failed: {e}")
                self.report["failed_augmentations"] += 1
                failed_attempts += 1
                continue
            
            aug_img = transformed.get("image")
            aug_bboxes = transformed.get("bboxes", [])
            aug_labels = transformed.get("class_labels", [])
            
            # Skip if no valid boxes after augmentation
            if not aug_bboxes:
                failed_attempts += 1
                continue
            
            # Validate augmented image quality
            aug_quality = validate_image_quality(aug_img)
            if not aug_quality['is_valid']:
                failed_attempts += 1
                continue
            
            # Process and validate final boxes
            final_boxes = []
            for cls, bbox in zip(aug_labels, aug_bboxes):
                x_c, y_c, bw, bh = bbox
                # Clamp to valid range
                x_c = float(np.clip(x_c, 0.0, 1.0))
                y_c = float(np.clip(y_c, 0.0, 1.0))
                bw = float(np.clip(bw, 1e-6, 1.0))
                bh = float(np.clip(bh, 1e-6, 1.0))
                final_boxes.append([int(cls), x_c, y_c, bw, bh])
            
            if not normalized_boxes_ok(final_boxes):
                failed_attempts += 1
                continue
            
            # Save augmented image and label
            out_name = f"{img_path.stem}_aug_{augmented_count:05d}{img_path.suffix}"
            out_img_path = dst_images / out_name
            out_lbl_path = dst_labels / f"{Path(out_name).stem}.txt"
            
            try:
                success = cv2.imwrite(str(out_img_path), aug_img)
                if not success:
                    raise Exception("Failed to write image")
                save_yolo_label(out_lbl_path, final_boxes)
            except Exception as e:
                logger.debug(f"⚠️ Failed to save: {e}")
                # Cleanup
                if out_img_path.exists():
                    out_img_path.unlink(missing_ok=True)
                if out_lbl_path.exists():
                    out_lbl_path.unlink(missing_ok=True)
                failed_attempts += 1
                continue
            
            # Update counters and reports
            augmented_count += 1
            self.report["pipeline_usage"][pipeline_name] += 1
            self.report["quality_metrics"]["blur_scores"].append(aug_quality['blur_score'])
            self.report["quality_metrics"]["brightness"].append(aug_quality['brightness'])
            self.report["quality_metrics"]["contrast"].append(aug_quality['contrast'])
            
            pbar.update(1)
            
            if augmented_count % 100 == 0:
                pbar.set_postfix({
                    "augmented": augmented_count,
                    "failed": failed_attempts,
                    "pipeline": pipeline_name[:10]
                })
        
        pbar.close()
        
        if failed_attempts >= max_failed_attempts:
            logger.warning(f"⚠️ Too many failed attempts for {split}, stopping early")
        
        # Final split statistics
        self.report["splits"][split] = {
            "original_count": original_count,
            "skipped_originals": skipped_originals,
            "copied_originals": copied_originals,
            "augmented_count": augmented_count,
            "failed_attempts": failed_attempts,
            "final_total": copied_originals + augmented_count,
            "augmentation_ratio": augmented_count / max(1, copied_originals),
            "success_rate": augmented_count / max(1, augmented_count + failed_attempts)
        }
        
        logger.info(f"✅ {split.upper()} complete: {copied_originals} originals + {augmented_count} augmented = {copied_originals + augmented_count} total")
        logger.info(f"   Success rate: {(augmented_count/max(1, augmented_count + failed_attempts))*100:.1f}%")

    def _calculate_quality_statistics(self):
        """Calculate quality metrics from collected data"""
        quality = self.report["quality_metrics"]
        
        if quality.get("blur_scores"):
            scores = quality["blur_scores"]
            self.report["quality_statistics"] = {
                "blur": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "samples": len(scores)
                },
                "brightness": {
                    "mean": float(np.mean(quality.get("brightness", [0]))),
                    "std": float(np.std(quality.get("brightness", [0])))
                },
                "contrast": {
                    "mean": float(np.mean(quality.get("contrast", [0]))),
                    "std": float(np.std(quality.get("contrast", [0])))
                }
            }

    def _write_data_yaml(self):
        """Create comprehensive data.yaml"""
        data_yaml = {
            "path": str(self.augmented_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": int(self.config.get("dataset", {}).get("nc", 1)),
            "names": self.config.get("dataset", {}).get("names", ["license_plate"]),
            "metadata": {
                "augmentation_version": "v2.0_fixed",
                "seed": self.seed,
                "multiplier": self.multiplier,
                "albu_version": A.__version__,
                "creation_date": datetime.now().isoformat(),
                "note": "Created with Ultimate Augmentation Pipeline v2.0 (Fixed API)"
            }
        }
        
        with open(self.augmented_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, sort_keys=False)
        
        logger.info(f"📄 data.yaml created: {self.augmented_dir / 'data.yaml'}")

    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("ULTIMATE AUGMENTATION PIPELINE - COMPLETE SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        total_original = sum(split.get("original_count", 0) for split in self.report["splits"].values())
        total_augmented = sum(split.get("augmented_count", 0) for split in self.report["splits"].values())
        total_final = sum(split.get("final_total", 0) for split in self.report["splits"].values())
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Original dataset: {total_original:,} images")
        print(f"   Augmented images created: {total_augmented:,}")
        print(f"   Final dataset size: {total_final:,}")
        print(f"   Augmentation ratio: {total_augmented/max(1,total_original):.2f}x")
        print(f"   Failed augmentations: {self.report.get('failed_augmentations', 0):,}")
        print(f"   Invalid images skipped: {self.report.get('invalid_images_skipped', 0):,}")
        print(f"   Albumentations version: {self.report.get('albu_version', 'unknown')}")
        
        # Pipeline usage
        print(f"\n🎨 PIPELINE USAGE:")
        pipeline_usage = self.report.get("pipeline_usage", {})
        total_usage = sum(pipeline_usage.values())
        
        for pipeline, count in sorted(pipeline_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_usage * 100 if total_usage > 0 else 0
            print(f"   {pipeline.replace('_', ' ').title():20s}: {count:6d} ({percentage:.1f}%)")
        
        # Split details
        print(f"\n📈 SPLIT-WISE RESULTS:")
        for split, stats in self.report.get("splits", {}).items():
            print(f"\n   {split.upper():6s}:")
            print(f"     Original images: {stats.get('original_count', 0):,}")
            print(f"     Copied originals: {stats.get('copied_originals', 0):,}")
            print(f"     Augmented images: {stats.get('augmented_count', 0):,}")
            print(f"     Final total: {stats.get('final_total', 0):,}")
            
            if stats.get('copied_originals', 0) > 0:
                ratio = stats.get('augmented_count', 0) / stats.get('copied_originals', 0)
                print(f"     Augmentation ratio: {ratio:.2f}x")
            
            if stats.get('success_rate', 0) > 0:
                print(f"     Success rate: {stats.get('success_rate', 0)*100:.1f}%")
        
        # Quality statistics
        if "quality_statistics" in self.report:
            print(f"\n🔍 QUALITY METRICS:")
            q_stats = self.report["quality_statistics"]
            
            if "blur" in q_stats:
                blur = q_stats["blur"]
                print(f"   Blur score: {blur.get('mean', 0):.1f} ± {blur.get('std', 0):.1f}")
                print(f"     Range: {blur.get('min', 0):.1f} - {blur.get('max', 0):.1f}")
        
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"   Augmented dataset: {self.augmented_dir}")
        print(f"   Augmentation report: {self.augmented_dir / 'ultimate_augmentation_report_v2.json'}")
        print(f"   Configuration file: {self.augmented_dir / 'data.yaml'}")
        
        print("\n✅ AUGMENTATION COMPLETE - READY FOR PHASE 1 TRAINING!")
        print("=" * 70)


# ---------------------------
# Check and fix albumentations API (ĐÃ SỬA)
# ---------------------------
def check_albu_version():
    """Check albumentations version and API compatibility"""
    import albumentations as A
    version = A.__version__
    print(f"📦 Albumentations version: {version}")
    
    # Major version check
    major_version = int(version.split('.')[0]) if '.' in version else 0
    
    if major_version >= 2:
        print("✅ Using Albumentations 2.x API")
        # Test 2.x API
        test_albu_api()
    else:
        print("⚠️ Using Albumentations 1.x or older API")
    
    return version


# ---------------------------
# Main execution (ĐÃ SỬA)
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Ultimate Dataset Augmenter v2.0 (Fixed)")
    parser.add_argument("--config", type=str, default="configs/data_license_plate.yaml", 
                       help="Path to config YAML")
    parser.add_argument("--multiplier", type=float, default=2.0, 
                       help="Target dataset size multiplier (>=1.0)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--preview", type=str, default=None, 
                       help="Path to image for preview (no batch processing)")
    parser.add_argument("--n-previews", type=int, default=3, 
                       help="Number of preview images")
    
    args = parser.parse_args()
    
    # THÊM DÒNG NÀY ĐỂ BỎ QUA CẢNH BÁO
    warnings.filterwarnings('ignore', message='.*are not valid for transform.*')
    
    # Check albumentations version
    albu_version = check_albu_version()
    
    # If old version, use simple pipelines
    if albu_version.startswith("0.") or albu_version.startswith("1.0"):
        print("⚠️ Older albumentations version detected, using simplified pipelines")
    
    augmenter = UltimateDatasetAugmenter(args.config, args.multiplier, args.seed)
    
    if args.preview:
        # Preview mode
        preview_path = Path(args.preview)
        if not preview_path.exists():
            logger.error(f"Preview image not found: {preview_path}")
            return
        
        # Generate previews
        out_dir = Path("results/augmentation_previews_v2")
        ensure_dir(out_dir)
        
        image = cv2.imread(str(preview_path))
        if image is None:
            logger.error(f"Cannot read image: {preview_path}")
            return
        
        # Apply each pipeline
        for i, pname in enumerate(augmenter.pipeline_names[:args.n_previews]):
            pipeline = augmenter.pipelines[pname]
            
            # Create dummy bbox
            h, w = image.shape[:2]
            dummy_boxes = [[0, 0.5, 0.5, 0.3, 0.1]]
            
            try:
                transformed = pipeline(
                    image=image.copy(),
                    bboxes=[b[1:] for b in dummy_boxes],
                    class_labels=[int(b[0]) for b in dummy_boxes]
                )
                
                aug_img = transformed.get("image")
                if aug_img is not None:
                    out_path = out_dir / f"{preview_path.stem}_{pname}{preview_path.suffix}"
                    cv2.imwrite(str(out_path), aug_img)
                    logger.info(f"✅ Saved preview: {out_path}")
                else:
                    logger.warning(f"⚠️ No output from pipeline: {pname}")
                    
            except Exception as e:
                logger.error(f"❌ Pipeline {pname} failed: {e}")
        
        logger.info(f"📁 Preview generation complete: {out_dir}")
        return
    
    # Run full pipeline
    try:
        augmenter.augment_all_splits()
        print("\n🎉 Augmentation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Augmentation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()