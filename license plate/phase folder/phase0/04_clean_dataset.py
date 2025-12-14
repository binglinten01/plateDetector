#!/usr/bin/env python3
"""
Ultimate Dataset Cleaning Pipeline with Auto-Detection and Quality Control
Senior-Level cleaning with multiple validation checks and smart filtering
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger
from src.utils.image_utils import (
    verify_image, calculate_blur_score,
    calculate_brightness_contrast, perspective_transform
)
from src.utils.detection_utils import (
    convert_bbox_format, calculate_iou,
    clip_bbox, nms
)

logger = setup_logger("DatasetCleaner")

class UltimateDatasetCleaner:
    """Ultimate dataset cleaner with advanced quality control."""
    
    def __init__(self, config_path: str = "configs/data_license_plate.yaml"):
        """Initialize cleaner with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.processed_dir = Path(self.config['paths']['processed_data'])
        self.cleaned_dir = Path(self.config['paths']['cleaned_data'])
        self.flagged_dir = Path(self.config['paths']['flagged_data'])
        
        # Create directories
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        
        # Create flagged subdirectories
        flagged_categories = ['blurry', 'mislabeled', 'corrupted', 
                             'duplicate', 'invalid_bbox', 'poor_quality',
                             'size_issues', 'aspect_ratio_issues']
        
        for category in flagged_categories:
            (self.flagged_dir / category).mkdir(parents=True, exist_ok=True)
        
        # Cleaning thresholds
        self.blur_threshold = self.config['cleaning']['blur_threshold_laplacian']
        self.min_bbox_area = self.config['cleaning']['min_bbox_area_px']
        self.max_bbox_area = self.config['cleaning'].get('max_bbox_area_px', float('inf'))
        self.iou_threshold = self.config['cleaning']['iou_label_detector_threshold']
        
        # Additional cleaning parameters
        self.min_aspect_ratio = 1.5  # Minimum aspect ratio for license plates
        self.max_aspect_ratio = 6.0   # Maximum aspect ratio for license plates
        self.min_brightness = 30      # Minimum acceptable brightness
        self.max_brightness = 220     # Maximum acceptable brightness
        self.min_contrast = 20        # Minimum acceptable contrast
        
        # Load small detector if available (for mislabel detection)
        self.detector = None
        self._load_small_detector()
        
        logger.info(f"Initialized UltimateDatasetCleaner")
        logger.info(f"  Blur threshold: {self.blur_threshold}")
        logger.info(f"  BBox area range: {self.min_bbox_area} - {self.max_bbox_area}")
        logger.info(f"  IoU threshold: {self.iou_threshold}")
    
    def _load_small_detector(self):
        """Load a small pre-trained detector for label validation."""
        try:
            # Try to load a small YOLO model for validation
            import torch
            from ultralytics import YOLO
            
            model_path = "models/pretrained/yolo11n.pt"
            if Path(model_path).exists():
                self.detector = YOLO(model_path)
                logger.info(f"Loaded small detector from: {model_path}")
            else:
                logger.warning(f"Small detector not found at: {model_path}")
                logger.info("Label validation will be limited to geometric checks")
                
        except ImportError:
            logger.warning("Ultralytics YOLO not installed. Install with: pip install ultralytics")
        except Exception as e:
            logger.warning(f"Could not load detector: {e}")
    
    def clean_dataset(self) -> Dict:
        """Clean the entire dataset with comprehensive quality checks.
        
        Returns:
            Detailed cleaning report
        """
        logger.info("=" * 60)
        logger.info("STARTING ULTIMATE DATASET CLEANING PIPELINE")
        logger.info("=" * 60)
        
        cleaning_report = {
            'timestamp': datetime.now().isoformat(),
            'total_processed': 0,
            'cleaned_count': 0,
            'flagged_count': 0,
            'removed_count': 0,
            'flagged_by_category': defaultdict(int),
            'issues_fixed': defaultdict(int),
            'quality_metrics': {},
            'split_reports': {}
        }
        
        # Process each split
        splits = ['train', 'val', 'test']
        
        for split in splits:
            logger.info(f"\n🔧 Cleaning {split.upper()} split...")
            
            split_report = self._clean_split(split)
            cleaning_report['split_reports'][split] = split_report
            
            # Update overall report
            cleaning_report['total_processed'] += split_report.get('total_processed', 0)
            cleaning_report['cleaned_count'] += split_report.get('cleaned_count', 0)
            cleaning_report['flagged_count'] += split_report.get('flagged_count', 0)
            cleaning_report['removed_count'] += split_report.get('removed_count', 0)
            
            for category, count in split_report.get('flagged_by_category', {}).items():
                cleaning_report['flagged_by_category'][category] += count
            
            for issue, count in split_report.get('issues_fixed', {}).items():
                cleaning_report['issues_fixed'][issue] += count
        
        # Calculate overall quality metrics
        if cleaning_report['total_processed'] > 0:
            cleaning_report['quality_metrics'] = {
                'cleaning_rate': cleaning_report['cleaned_count'] / cleaning_report['total_processed'] * 100,
                'flagging_rate': cleaning_report['flagged_count'] / cleaning_report['total_processed'] * 100,
                'average_issues_per_image': sum(cleaning_report['flagged_by_category'].values()) / max(1, cleaning_report['total_processed']),
                'most_common_issue': max(cleaning_report['flagged_by_category'].items(), key=lambda x: x[1])[0] if cleaning_report['flagged_by_category'] else 'None'
            }
        
        # Save comprehensive cleaning report
        self._save_cleaning_report(cleaning_report)
        
        # Create cleaned dataset structure
        self._create_cleaned_dataset_structure()
        
        # Print summary
        self._print_cleaning_summary(cleaning_report)
        
        return dict(cleaning_report)
    
    def _clean_split(self, split: str) -> Dict:
        """Clean a specific dataset split with multiple validation checks.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Split cleaning report
        """
        split_report = {
            'split': split,
            'total_processed': 0,
            'cleaned_count': 0,
            'flagged_count': 0,
            'removed_count': 0,
            'flagged_by_category': defaultdict(int),
            'issues_fixed': defaultdict(int),
            'per_image_reports': []
        }
        
        # Get images and labels for this split
        images_dir = self.processed_dir / "images" / split
        labels_dir = self.processed_dir / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Split {split} not found or incomplete")
            return split_report
        
        # Create cleaned directories for this split
        cleaned_images_dir = self.cleaned_dir / "images" / split
        cleaned_labels_dir = self.cleaned_dir / "labels" / split
        cleaned_images_dir.mkdir(parents=True, exist_ok=True)
        cleaned_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        split_report['total_processed'] = len(image_files)
        
        if not image_files:
            return split_report
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Cleaning {split}", unit="image"):
            try:
                # Get corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Perform comprehensive checks
                check_result = self._comprehensive_image_check(img_path, label_path)
                
                image_report = {
                    'image': img_path.name,
                    'checks_passed': check_result['checks_passed'],
                    'issues_found': check_result['issues'],
                    'issues_fixed': check_result['fixed_issues'],
                    'should_keep': check_result['should_keep']
                }
                
                split_report['per_image_reports'].append(image_report)
                
                if check_result['should_keep']:
                    # Copy to cleaned directory
                    shutil.copy2(img_path, cleaned_images_dir / img_path.name)
                    
                    # Apply fixes to label if needed
                    if label_path.exists():
                        self._apply_label_fixes(label_path, cleaned_labels_dir, check_result)
                    
                    split_report['cleaned_count'] += 1
                    
                    # Record fixed issues
                    for issue in check_result['fixed_issues']:
                        split_report['issues_fixed'][issue] += 1
                else:
                    # Flag for review
                    self._flag_for_review(img_path, label_path, check_result['issues'])
                    split_report['flagged_count'] += 1
                    
                    # Record flagged categories
                    for issue in check_result['issues']:
                        split_report['flagged_by_category'][issue] += 1
                    
                    split_report['removed_count'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                # Flag as processing error
                self._flag_for_review(img_path, None, ['processing_error'])
                split_report['flagged_count'] += 1
                split_report['flagged_by_category']['processing_error'] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        split_report['flagged_by_category'] = dict(split_report['flagged_by_category'])
        split_report['issues_fixed'] = dict(split_report['issues_fixed'])
        
        # Keep only a sample of per-image reports to avoid huge files
        if len(split_report['per_image_reports']) > 100:
            import random
            random.seed(42)
            split_report['per_image_reports'] = random.sample(split_report['per_image_reports'], 100)
        
        return split_report
    
    def _comprehensive_image_check(self, img_path: Path, label_path: Optional[Path]) -> Dict:
        """Perform comprehensive checks on image and label.
        
        Args:
            img_path: Path to image
            label_path: Path to label file
            
        Returns:
            Dictionary with check results
        """
        issues = []
        fixed_issues = []
        checks_passed = []
        
        # 1. Basic file validation
        if not verify_image(str(img_path)):
            issues.append('corrupted_image')
            return {
                'checks_passed': checks_passed,
                'issues': issues,
                'fixed_issues': fixed_issues,
                'should_keep': False
            }
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append('unreadable_image')
            return {
                'checks_passed': checks_passed,
                'issues': issues,
                'fixed_issues': fixed_issues,
                'should_keep': False
            }
        
        h, w = img.shape[:2]
        checks_passed.append('image_readable')
        
        # 2. Image quality checks
        # Blur check
        blur_score = calculate_blur_score(img)
        if blur_score < self.blur_threshold:
            issues.append('blurry')
        else:
            checks_passed.append('acceptable_blur')
        
        # Brightness and contrast check
        brightness, contrast = calculate_brightness_contrast(img)
        if brightness < self.min_brightness:
            issues.append('too_dark')
        elif brightness > self.max_brightness:
            issues.append('too_bright')
        else:
            checks_passed.append('acceptable_brightness')
        
        if contrast < self.min_contrast:
            issues.append('low_contrast')
        else:
            checks_passed.append('acceptable_contrast')
        
        # Check for uniform color (all black or all white)
        if np.all(img == 0) or np.all(img == 255):
            issues.append('uniform_color')
        
        # 3. Label validation
        if label_path is None or not label_path.exists():
            issues.append('missing_label')
            return {
                'checks_passed': checks_passed,
                'issues': issues,
                'fixed_issues': fixed_issues,
                'should_keep': False
            }
        
        checks_passed.append('label_exists')
        
        # Read and validate labels
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except:
            issues.append('corrupted_label')
            return {
                'checks_passed': checks_passed,
                'issues': issues,
                'fixed_issues': fixed_issues,
                'should_keep': False
            }
        
        valid_annotations = []
        has_label_issues = False
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                issues.append('invalid_label_format')
                has_label_issues = True
                continue
            
            try:
                class_id = int(parts[0])
                yolo_coords = list(map(float, parts[1:5]))
                
                # Check if coordinates are valid (normalized 0-1)
                if any(c < 0 or c > 1 for c in yolo_coords):
                    issues.append('invalid_coordinates')
                    has_label_issues = True
                    continue
                
                # Convert to absolute coordinates for validation
                abs_bbox = convert_bbox_format(yolo_coords, 'yolo', 'xyxy', (w, h))
                x1, y1, x2, y2 = abs_bbox
                
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Check bbox size
                if area < self.min_bbox_area:
                    issues.append('bbox_too_small')
                    has_label_issues = True
                    continue
                
                if area > self.max_bbox_area:
                    issues.append('bbox_too_large')
                    has_label_issues = True
                    continue
                
                # Check aspect ratio
                if height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                        issues.append('unusual_aspect_ratio')
                        # Don't fail for this, just note it
                
                # Clip bbox if needed
                clipped_coords = clip_bbox(yolo_coords, (w, h), 'yolo')
                if yolo_coords != clipped_coords:
                    # Bbox was clipped, update coordinates
                    yolo_coords = clipped_coords
                    fixed_issues.append('clipped_bbox')
                
                valid_annotations.append({
                    'class_id': class_id,
                    'yolo_coords': yolo_coords,
                    'abs_bbox': abs_bbox,
                    'area': area,
                    'aspect_ratio': aspect_ratio if height > 0 else 0
                })
                
            except Exception as e:
                issues.append('label_parsing_error')
                has_label_issues = True
                continue
        
        # Check for duplicate bounding boxes
        if len(valid_annotations) > 1:
            bboxes = [ann['abs_bbox'] for ann in valid_annotations]
            scores = [1.0] * len(bboxes)  # Dummy scores
            
            # Use NMS to find duplicates
            keep_indices = nms(bboxes, scores, iou_threshold=0.9, format='xyxy')
            
            if len(keep_indices) < len(bboxes):
                issues.append('duplicate_bbox')
                # Keep only non-duplicate annotations
                valid_annotations = [valid_annotations[i] for i in keep_indices]
                fixed_issues.append('removed_duplicates')
        
        # Check if image has at least one valid license plate
        if not valid_annotations:
            issues.append('no_valid_plates')
            has_label_issues = True
        
        checks_passed.append('valid_annotations')
        
        # 4. Optional: Validate with small detector
        if self.detector and not has_label_issues:
            try:
                detector_validation = self._validate_with_detector(img, valid_annotations, (w, h))
                if not detector_validation['is_valid']:
                    issues.append('mislabeled')
                    issues.extend(detector_validation.get('details', []))
            except:
                pass  # Skip detector validation if it fails
        
        # 5. Determine whether to keep the image
        # Critical issues that cause rejection
        critical_issues = {'corrupted_image', 'unreadable_image', 'missing_label', 
                          'corrupted_label', 'no_valid_plates', 'invalid_label_format'}
        
        has_critical_issues = any(issue in critical_issues for issue in issues)
        
        # Too many issues also cause rejection
        total_issues = len(issues)
        should_keep = not has_critical_issues and total_issues <= 3
        
        return {
            'checks_passed': checks_passed,
            'issues': issues,
            'fixed_issues': fixed_issues,
            'should_keep': should_keep,
            'valid_annotations': valid_annotations,
            'image_metrics': {
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'width': w,
                'height': h
            }
        }
    
    def _validate_with_detector(self, image: np.ndarray, annotations: List[Dict], image_size: Tuple[int, int]) -> Dict:
        """Validate annotations using a small pre-trained detector.
        
        Args:
            image: Input image
            annotations: List of annotations
            image_size: (width, height) of image
            
        Returns:
            Validation results
        """
        try:
            # Run detector
            results = self.detector(image, verbose=False)
            
            if not results or len(results) == 0:
                return {'is_valid': True, 'details': []}
            
            # Get detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        if conf > 0.3:  # Confidence threshold
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf)
                            })
            
            if not detections:
                return {'is_valid': True, 'details': []}
            
            # Match annotations with detections
            matched = 0
            details = []
            
            for ann in annotations:
                ann_bbox = ann['abs_bbox']
                best_iou = 0
                
                for det in detections:
                    det_bbox = det['bbox']
                    iou = calculate_iou(ann_bbox, det_bbox, 'xyxy')
                    best_iou = max(best_iou, iou)
                
                if best_iou > self.iou_threshold:
                    matched += 1
                else:
                    details.append(f'low_iou_{best_iou:.2f}')
            
            # Check if enough annotations are matched
            match_ratio = matched / len(annotations) if annotations else 0
            
            return {
                'is_valid': match_ratio >= 0.5,  # At least 50% should match
                'details': details,
                'match_ratio': match_ratio,
                'matched': matched,
                'total': len(annotations)
            }
            
        except Exception as e:
            logger.debug(f"Detector validation failed: {e}")
            return {'is_valid': True, 'details': ['detector_failed']}
    
    def _apply_label_fixes(self, original_label_path: Path, output_dir: Path, check_result: Dict):
        """Apply fixes to label file based on check results.
        
        Args:
            original_label_path: Original label file
            output_dir: Output directory for fixed label
            check_result: Check results dictionary
        """
        try:
            output_path = output_dir / original_label_path.name
            
            if 'valid_annotations' in check_result and check_result['valid_annotations']:
                # Write fixed annotations
                with open(output_path, 'w') as f:
                    for ann in check_result['valid_annotations']:
                        coords = ann['yolo_coords']
                        f.write(f"{ann['class_id']} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
            else:
                # Copy original if no fixes needed
                shutil.copy2(original_label_path, output_path)
                
        except Exception as e:
            logger.error(f"Error applying label fixes: {e}")
            # Fallback: copy original
            shutil.copy2(original_label_path, output_dir / original_label_path.name)
    
    def _flag_for_review(self, img_path: Path, label_path: Optional[Path], issues: List[str]):
        """Flag image for manual review with detailed issue report.
        
        Args:
            img_path: Image file path
            label_path: Label file path (optional)
            issues: List of issues detected
        """
        # Determine primary category based on issues
        if 'corrupted_image' in issues or 'unreadable_image' in issues:
            category = 'corrupted'
        elif 'blurry' in issues:
            category = 'blurry'
        elif 'missing_label' in issues or 'corrupted_label' in issues or 'mislabeled' in issues:
            category = 'mislabeled'
        elif 'bbox_too_small' in issues or 'bbox_too_large' in issues:
            category = 'size_issues'
        elif 'unusual_aspect_ratio' in issues:
            category = 'aspect_ratio_issues'
        elif 'too_dark' in issues or 'too_bright' in issues or 'low_contrast' in issues:
            category = 'poor_quality'
        else:
            category = 'invalid_bbox'
        
        # Create destination directory
        dest_dir = self.flagged_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        shutil.copy2(img_path, dest_dir / img_path.name)
        
        # Copy label if exists
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_dir / label_path.name)
        
        # Create detailed issue report
        report = {
            'image': img_path.name,
            'issues': issues,
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'original_location': str(img_path)
        }
        
        report_path = dest_dir / f"{img_path.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _save_cleaning_report(self, cleaning_report: Dict):
        """Save comprehensive cleaning report.
        
        Args:
            cleaning_report: Cleaning report dictionary
        """
        report_path = self.cleaned_dir / "ultimate_cleaning_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(cleaning_report, f, indent=2)
        
        logger.info(f"📄 Cleaning report saved to: {report_path}")
    
    def _create_cleaned_dataset_structure(self):
        """Create cleaned dataset structure with data.yaml."""
        # Create data.yaml for YOLO
        data_yaml = {
            'path': str(self.cleaned_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.config['dataset']['classes']),
            'names': self.config['dataset']['classes'],
            'cleaning_info': {
                'blur_threshold': self.blur_threshold,
                'min_bbox_area': self.min_bbox_area,
                'max_bbox_area': self.max_bbox_area,
                'iou_threshold': self.iou_threshold,
                'cleaning_timestamp': datetime.now().isoformat()
            }
        }
        
        yaml_path = self.cleaned_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:  # <-- THÊM encoding='utf-8' Ở ĐÂY NỮA
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Create README for cleaned dataset
        readme_content = f"""# Cleaned License Plate Dataset

    ## Overview
    This dataset has been cleaned using the Ultimate Dataset Cleaning Pipeline.

    ## Cleaning Parameters
    - Blur threshold (Laplacian variance): {self.blur_threshold}
    - Minimum bounding box area: {self.min_bbox_area} pixels
    - Maximum bounding box area: {self.max_bbox_area} pixels
    - IoU threshold for label validation: {self.iou_threshold}

    ## Directory Structure 
    datasets/03_cleaned/
    ├── images/
    │ ├── train/
    │ ├── val/
    │ └── test/
    ├── labels/
    │ ├── train/
    │ ├── val/
    │ └── test/
    ├── data.yaml
    └── ultimate_cleaning_report.json

    ## Usage
    Use `data.yaml` for YOLO training.

    ## Notes
    - Flagged images are available in `{self.flagged_dir}`
    - See `ultimate_cleaning_report.json` for detailed cleaning statistics
    """
        
        readme_path = self.cleaned_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:  # <-- SỬA DÒNG NÀY
            f.write(readme_content)
        
        logger.info(f"📁 Cleaned dataset structure created at: {self.cleaned_dir}")
    
    def _print_cleaning_summary(self, cleaning_report: Dict):
        """Print cleaning summary.
        
        Args:
            cleaning_report: Cleaning report dictionary
        """
        print("\n" + "=" * 70)
        print("DATASET CLEANING - COMPLETE SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 PROCESSING STATISTICS")
        print(f"   Total images processed: {cleaning_report['total_processed']:,}")
        print(f"   Images kept (cleaned): {cleaning_report['cleaned_count']:,} ({cleaning_report['cleaned_count']/cleaning_report['total_processed']*100:.1f}%)")
        print(f"   Images flagged for review: {cleaning_report['flagged_count']:,} ({cleaning_report['flagged_count']/cleaning_report['total_processed']*100:.1f}%)")
        print(f"   Images removed: {cleaning_report['removed_count']:,}")
        
        print(f"\n🚩 FLAGGED IMAGES BY CATEGORY")
        flagged_by_category = cleaning_report.get('flagged_by_category', {})
        if flagged_by_category:
            for category, count in sorted(flagged_by_category.items(), key=lambda x: x[1], reverse=True):
                percentage = count / cleaning_report['flagged_count'] * 100 if cleaning_report['flagged_count'] > 0 else 0
                print(f"   {category.replace('_', ' ').title():25s}: {count:5d} ({percentage:.1f}%)")
        else:
            print("   No images were flagged!")
        
        print(f"\n🔧 ISSUES FIXED")
        issues_fixed = cleaning_report.get('issues_fixed', {})
        if issues_fixed:
            for issue, count in sorted(issues_fixed.items(), key=lambda x: x[1], reverse=True):
                print(f"   {issue.replace('_', ' ').title():25s}: {count:5d}")
        else:
            print("   No issues were fixed")
        
        print(f"\n📈 QUALITY METRICS")
        quality_metrics = cleaning_report.get('quality_metrics', {})
        if quality_metrics:
            print(f"   Cleaning rate: {quality_metrics.get('cleaning_rate', 0):.1f}%")
            print(f"   Flagging rate: {quality_metrics.get('flagging_rate', 0):.1f}%")
            print(f"   Avg issues per image: {quality_metrics.get('average_issues_per_image', 0):.2f}")
            print(f"   Most common issue: {quality_metrics.get('most_common_issue', 'None').replace('_', ' ').title()}")
        
        print(f"\n📁 OUTPUT LOCATIONS")
        print(f"   Cleaned dataset: {self.cleaned_dir}")
        print(f"   Flagged images: {self.flagged_dir}")
        print(f"   Cleaning report: {self.cleaned_dir / 'ultimate_cleaning_report.json'}")
        
        print("\n💡 NEXT STEPS")
        print("   1. Review flagged images in the flagged directory")
        print("   2. Use cleaned dataset for training")
        print("   3. Consider augmenting the cleaned dataset")
        
        print("=" * 70)
    
    def find_duplicate_images(self, method: str = 'phash', similarity_threshold: float = 0.95) -> List[List[str]]:
        """Find duplicate or near-duplicate images in the dataset.
        
        Args:
            method: Method for duplicate detection ('phash', 'md5', 'dhash')
            similarity_threshold: Similarity threshold (0-1)
            
        Returns:
            List of duplicate groups
        """
        logger.info(f"Finding duplicate images using {method} method...")
        
        # Collect all images
        all_images = []
        for split in ['train', 'val', 'test']:
            images_dir = self.processed_dir / "images" / split
            if images_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_images.extend(images_dir.glob(f"*{ext}"))
        
        if not all_images:
            logger.warning("No images found for duplicate detection")
            return []
        
        # Calculate hashes
        hashes = {}
        hash_to_images = defaultdict(list)
        
        for img_path in tqdm(all_images, desc="Calculating hashes"):
            try:
                if method == 'md5':
                    img_hash = self._calculate_md5_hash(img_path)
                elif method == 'phash':
                    img_hash = self._calculate_perceptual_hash(img_path)
                elif method == 'dhash':
                    img_hash = self._calculate_difference_hash(img_path)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if img_hash:
                    hashes[str(img_path)] = img_hash
                    hash_to_images[img_hash].append(str(img_path))
                    
            except Exception as e:
                logger.warning(f"Error calculating hash for {img_path}: {e}")
        
        # Find duplicates (same hash)
        exact_duplicates = [paths for paths in hash_to_images.values() if len(paths) > 1]
        
        # Find near-duplicates (similar hashes) for perceptual hashes
        near_duplicates = []
        if method in ['phash', 'dhash']:
            near_duplicates = self._find_near_duplicates(hashes, similarity_threshold)
        
        all_duplicates = exact_duplicates + near_duplicates
        
        # Save duplicate report
        report = {
            'method': method,
            'similarity_threshold': similarity_threshold,
            'total_images': len(all_images),
            'exact_duplicate_groups': len(exact_duplicates),
            'near_duplicate_groups': len(near_duplicates),
            'total_duplicate_groups': len(all_duplicates),
            'duplicate_groups': all_duplicates,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = self.flagged_dir / "duplicate_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Found {len(exact_duplicates)} exact duplicate groups and {len(near_duplicates)} near-duplicate groups")
        
        return all_duplicates
    
    def _calculate_md5_hash(self, img_path: Path) -> str:
        """Calculate MD5 hash of image file.
        
        Args:
            img_path: Path to image
            
        Returns:
            MD5 hash string
        """
        with open(img_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        
        return file_hash.hexdigest()
    
    def _calculate_perceptual_hash(self, img_path: Path) -> Optional[str]:
        """Calculate perceptual hash (pHash) of image.
        
        Args:
            img_path: Path to image
            
        Returns:
            pHash string or None
        """
        try:
            import imagehash
            from PIL import Image
            
            img = Image.open(img_path)
            phash = imagehash.phash(img)
            
            return str(phash)
            
        except ImportError:
            logger.warning("imagehash not installed. Install with: pip install imagehash")
            return None
        except Exception as e:
            logger.warning(f"Error calculating pHash: {e}")
            return None
    
    def _calculate_difference_hash(self, img_path: Path) -> Optional[str]:
        """Calculate difference hash (dHash) of image.
        
        Args:
            img_path: Path to image
            
        Returns:
            dHash string or None
        """
        try:
            import imagehash
            from PIL import Image
            
            img = Image.open(img_path)
            dhash = imagehash.dhash(img)
            
            return str(dhash)
            
        except ImportError:
            logger.warning("imagehash not installed")
            return None
        except Exception as e:
            logger.warning(f"Error calculating dHash: {e}")
            return None
    
    def _find_near_duplicates(self, hashes: Dict[str, str], threshold: float = 0.95) -> List[List[str]]:
        """Find near-duplicate images based on hash similarity.
        
        Args:
            hashes: Dictionary of image paths to hashes
            threshold: Similarity threshold
            
        Returns:
            List of near-duplicate groups
        """
        try:
            import imagehash
            
            # Convert string hashes back to imagehash objects
            hash_objects = {}
            for path, hash_str in hashes.items():
                try:
                    hash_objects[path] = imagehash.hex_to_hash(hash_str)
                except:
                    continue
            
            # Find similar hashes
            paths = list(hash_objects.keys())
            similar_groups = []
            visited = set()
            
            for i, path1 in enumerate(paths):
                if path1 in visited:
                    continue
                
                group = [path1]
                hash1 = hash_objects[path1]
                
                for j, path2 in enumerate(paths[i+1:], i+1):
                    if path2 in visited:
                        continue
                    
                    hash2 = hash_objects[path2]
                    similarity = 1 - (hash1 - hash2) / 64.0  # pHash uses 64 bits
                    
                    if similarity >= threshold:
                        group.append(path2)
                        visited.add(path2)
                
                if len(group) > 1:
                    similar_groups.append(group)
                    visited.add(path1)
            
            return similar_groups
            
        except Exception as e:
            logger.warning(f"Error finding near duplicates: {e}")
            return []


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ultimate Dataset Cleaner for License Plate Recognition"
    )
    parser.add_argument(
        "--config", 
        default="configs/data_license_plate.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--find-duplicates", 
        action="store_true",
        help="Find duplicate images only (don't run full cleaning)"
    )
    parser.add_argument(
        "--duplicate-method", 
        choices=['md5', 'phash', 'dhash'],
        default='phash',
        help="Duplicate detection method"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for near-duplicate detection"
    )
    
    args = parser.parse_args()
    
    cleaner = UltimateDatasetCleaner(args.config)
    
    if args.find_duplicates:
        duplicates = cleaner.find_duplicate_images(
            method=args.duplicate_method,
            similarity_threshold=args.similarity_threshold
        )
        
        print(f"\nFound {len(duplicates)} duplicate/near-duplicate groups")
        
        # Show first few groups
        for i, group in enumerate(duplicates[:5]):
            print(f"\nGroup {i+1} ({len(group)} images):")
            for path in group[:3]:  # Show first 3
                print(f"  {Path(path).name}")
            if len(group) > 3:
                print(f"  ... and {len(group)-3} more")
        
        if len(duplicates) > 5:
            print(f"\n... and {len(duplicates)-5} more groups")
        
        print(f"\nDetailed report saved to: {cleaner.flagged_dir / 'duplicate_report.json'}")
    else:
        report = cleaner.clean_dataset()
        print(f"\nCleaning complete. Report saved to: {cleaner.cleaned_dir / 'ultimate_cleaning_report.json'}")


if __name__ == "__main__":
    main()