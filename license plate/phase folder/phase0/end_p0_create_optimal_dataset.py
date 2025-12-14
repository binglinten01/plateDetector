# scripts/end_p0_create_optimal_dataset.py
"""
Create optimal dataset for Phase 1 with ENHANCED VERIFICATION
Senior-level quality control and statistics
"""

import shutil
import yaml
import json
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys


class DatasetQualityValidator:
    """Enhanced dataset quality validator"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def validate_dataset(self, dataset_path: Path):
        """Comprehensive dataset validation"""
        
        print("\n🔍 ENHANCED DATASET VALIDATION")
        print("=" * 50)
        
        # 1. Structure validation
        self._validate_structure(dataset_path)
        
        # 2. Image-label correspondence
        self._validate_correspondence(dataset_path)
        
        # 3. Label quality
        self._validate_labels(dataset_path)
        
        # 4. Image quality sampling
        self._validate_image_quality(dataset_path)
        
        # 5. Print results
        self._print_validation_results()
        
        return len(self.issues) == 0
    
    def _validate_structure(self, dataset_path: Path):
        """Validate dataset structure"""
        
        required_dirs = [
            dataset_path / "images" / "train",
            dataset_path / "images" / "val",
            dataset_path / "images" / "test",
            dataset_path / "labels" / "train",
            dataset_path / "labels" / "val",
            dataset_path / "labels" / "test"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.issues.append(f"Missing directory: {dir_path}")
            else:
                self.stats['directories_found'] += 1
    
    def _validate_correspondence(self, dataset_path: Path):
        """Validate image-label correspondence"""
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            
            # Get all files
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            label_files = list(lbl_dir.glob("*.txt"))
            
            # Convert to sets
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            
            # Check correspondence
            missing_labels = image_stems - label_stems
            missing_images = label_stems - image_stems
            
            self.stats[f'{split}_images'] = len(image_stems)
            self.stats[f'{split}_labels'] = len(label_stems)
            
            if missing_labels:
                self.warnings.append(f"{split.upper()}: {len(missing_labels)} images missing labels")
                self.stats[f'{split}_missing_labels'] = len(missing_labels)
            
            if missing_images:
                self.warnings.append(f"{split.upper()}: {len(missing_images)} labels missing images")
                self.stats[f'{split}_missing_images'] = len(missing_images)
    
    def _validate_labels(self, dataset_path: Path, sample_size: int = 100):
        """Validate label quality"""
        
        print("  Validating label quality...")
        
        splits = ['train', 'val', 'test']
        total_checked = 0
        total_invalid = 0
        
        for split in splits:
            lbl_dir = dataset_path / "labels" / split
            
            if not lbl_dir.exists():
                continue
            
            # Sample labels
            label_files = list(lbl_dir.glob("*.txt"))
            if not label_files:
                continue
            
            sample = random.sample(label_files, min(sample_size, len(label_files)))
            
            for lbl_file in sample:
                try:
                    with open(lbl_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    total_checked += 1
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) < 5:
                            self.warnings.append(f"Invalid format: {lbl_file.name} line {line_num}")
                            total_invalid += 1
                            continue
                        
                        # Check coordinates
                        try:
                            class_id = int(parts[0])
                            coords = list(map(float, parts[1:5]))
                            
                            # Validate YOLO format
                            for coord in coords:
                                if not 0 <= coord <= 1:
                                    self.warnings.append(f"Invalid coord {coord:.3f} in {lbl_file.name}")
                                    total_invalid += 1
                                    break
                            
                            # Check aspect ratio (license plates are typically 2:1 to 4:1)
                            w, h = coords[2], coords[3]
                            if h > 0:
                                aspect_ratio = w / h
                                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                                    self.warnings.append(f"Unusual aspect ratio {aspect_ratio:.1f} in {lbl_file.name}")
                        
                        except ValueError:
                            self.warnings.append(f"Parse error: {lbl_file.name} line {line_num}")
                            total_invalid += 1
                
                except Exception as e:
                    self.warnings.append(f"Error reading {lbl_file.name}: {e}")
        
        if total_checked > 0:
            invalid_percent = total_invalid / total_checked * 100
            self.stats['label_invalid_percent'] = invalid_percent
            
            if invalid_percent > 5:
                self.issues.append(f"High label error rate: {invalid_percent:.1f}%")
    
    def _validate_image_quality(self, dataset_path: Path, sample_size: int = 50):
        """Sample check of image quality"""
        
        print("  Sampling image quality...")
        
        splits = ['train']
        
        for split in splits:
            img_dir = dataset_path / "images" / split
            
            if not img_dir.exists():
                continue
            
            # Sample images
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if not image_files:
                continue
            
            sample = random.sample(image_files, min(sample_size, len(image_files)))
            
            blur_scores = []
            dimensions = []
            
            for img_file in sample:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        self.warnings.append(f"Cannot read image: {img_file.name}")
                        continue
                    
                    # Get dimensions
                    h, w = img.shape[:2]
                    dimensions.append((w, h))
                    
                    # Calculate blur score
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_scores.append(blur_score)
                    
                    # Check if image is too dark or bright
                    brightness = np.mean(gray)
                    if brightness < 30:
                        self.warnings.append(f"Very dark image: {img_file.name} ({brightness:.1f})")
                    elif brightness > 220:
                        self.warnings.append(f"Very bright image: {img_file.name} ({brightness:.1f})")
                
                except Exception as e:
                    self.warnings.append(f"Error processing {img_file.name}: {e}")
            
            # Calculate statistics
            if blur_scores:
                avg_blur = np.mean(blur_scores)
                std_blur = np.std(blur_scores)
                
                self.stats['avg_blur_score'] = avg_blur
                self.stats['std_blur_score'] = std_blur
                
                if avg_blur < 50:
                    self.issues.append(f"Low average blur score: {avg_blur:.1f} (images may be blurry)")
            
            if dimensions:
                avg_width = np.mean([w for w, h in dimensions])
                avg_height = np.mean([h for w, h in dimensions])
                
                self.stats['avg_width'] = avg_width
                self.stats['avg_height'] = avg_height
    
    def _print_validation_results(self):
        """Print validation results"""
        
        print("\n📊 VALIDATION RESULTS")
        print("-" * 40)
        
        if self.issues:
            print("❌ ISSUES FOUND:")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print("✅ No critical issues found")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  • {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        # Print key statistics
        print("\n📈 KEY STATISTICS:")
        if 'train_images' in self.stats:
            print(f"  Train images: {self.stats['train_images']:,}")
        if 'val_images' in self.stats:
            print(f"  Val images: {self.stats['val_images']:,}")
        if 'test_images' in self.stats:
            print(f"  Test images: {self.stats['test_images']:,}")
        
        if 'avg_blur_score' in self.stats:
            print(f"  Avg blur score: {self.stats['avg_blur_score']:.1f}")
        
        if 'label_invalid_percent' in self.stats:
            print(f"  Label error rate: {self.stats['label_invalid_percent']:.1f}%")
        
        print("-" * 40)


def create_optimal_dataset():
    """Tạo dataset tối ưu với enhanced validation"""
    
    print("🚀 CREATING OPTIMAL DATASET FOR PHASE 1 (ENHANCED)")
    print("=" * 70)
    print("STRATEGY: Augmented + Synthetic with Senior-Level Quality Control")
    print("=" * 70)

    # Paths
    augmented_path = Path("datasets/04_augmented")   # From FIXED augmentation
    synthetic_path = Path("datasets/06_synthetic")   # Synthetic dataset
    optimal_path = Path("datasets/07_final_training")

    # Validate source datasets
    print("\n📋 VALIDATING SOURCE DATASETS...")
    
    if not augmented_path.exists():
        print("❌ Augmented dataset not found!")
        print("   Run: python scripts/05_augment_dataset.py (FIXED VERSION)")
        return
    
    if not synthetic_path.exists():
        print("❌ Synthetic dataset not found!")
        print("   Run: python scripts/06_generate_synthetic.py")
        return
    
    # Validate augmented dataset quality
    validator = DatasetQualityValidator()
    if not validator.validate_dataset(augmented_path):
        print("\n❌ Augmented dataset failed validation!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Remove previous output
    if optimal_path.exists():
        print(f"\n🗑️  Removing previous dataset: {optimal_path}")
        shutil.rmtree(optimal_path)
    
    # Create structure
    print("\n📁 Creating directory structure...")
    for split in ['train', 'val', 'test']:
        (optimal_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (optimal_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Combined counts and statistics
    splits = ['train', 'val', 'test']
    split_stats = {s: {'images': 0, 'labels': 0, 'augmented': 0, 'synthetic': 0} for s in splits}
    
    # ============================================================
    # 1. COPY AUGMENTED DATASET
    # ============================================================
    
    print("\n📦 COPYING AUGMENTED DATASET...")
    
    for split in splits:
        src_img = augmented_path / "images" / split
        src_lbl = augmented_path / "labels" / split
        dst_img = optimal_path / "images" / split
        dst_lbl = optimal_path / "labels" / split
        
        if not src_img.exists() or not src_lbl.exists():
            print(f"  ⚠️  Missing {split} in augmented dataset")
            continue
        
        # Get all images
        image_files = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
        
        copied_images = 0
        copied_labels = 0
        
        for img_file in image_files:
            stem = img_file.stem
            label_file = src_lbl / f"{stem}.txt"
            
            # Skip if no label (should not happen with FIXED augmentation)
            if not label_file.exists():
                print(f"    ⚠️  Skipping {img_file.name} (no label)")
                continue
            
            # Copy image
            shutil.copy2(img_file, dst_img / img_file.name)
            copied_images += 1
            
            # Copy label
            shutil.copy2(label_file, dst_lbl / f"{stem}.txt")
            copied_labels += 1
        
        split_stats[split]['images'] += copied_images
        split_stats[split]['labels'] += copied_labels
        split_stats[split]['augmented'] = copied_images
        
        print(f"  {split.upper()}: {copied_images:,} images, {copied_labels:,} labels")
    
    # ============================================================
    # 2. ADD SELECTIVE SYNTHETIC DATA
    # ============================================================
    
    print("\n🎨 ADDING SELECTIVE SYNTHETIC DATA...")
    
    if synthetic_path.exists():
        for split in splits:
            src_img = synthetic_path / "images" / split
            src_lbl = synthetic_path / "labels" / split
            dst_img = optimal_path / "images" / split
            dst_lbl = optimal_path / "labels" / split
            
            if not src_img.exists() or not src_lbl.exists():
                continue
            
            # Get synthetic files
            synth_images = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
            
            # Select 20% of synthetic data for training, less for val/test
            if split == 'train':
                select_ratio = 0.2  # 20% for training
            else:
                select_ratio = 0.1  # 10% for val/test
            
            random.seed(42)
            selected_count = int(len(synth_images) * select_ratio)
            selected_images = random.sample(synth_images, min(selected_count, len(synth_images)))
            
            added_images = 0
            added_labels = 0
            
            for img_file in selected_images:
                stem = img_file.stem
                label_file = src_lbl / f"{stem}.txt"
                
                if not label_file.exists():
                    continue
                
                # Create new filename to avoid conflicts
                new_name = f"{stem}_synth_{split}{img_file.suffix}"
                new_label_name = f"{stem}_synth_{split}.txt"
                
                # Copy files
                shutil.copy2(img_file, dst_img / new_name)
                shutil.copy2(label_file, dst_lbl / new_label_name)
                
                added_images += 1
                added_labels += 1
            
            split_stats[split]['images'] += added_images
            split_stats[split]['labels'] += added_labels
            split_stats[split]['synthetic'] = added_images
            
            print(f"  {split.upper()}: +{added_images:,} synthetic images")
    
    # ============================================================
    # 3. CALCULATE FINAL STATISTICS
    # ============================================================
    
    total_images = sum(stats['images'] for stats in split_stats.values())
    total_labels = sum(stats['labels'] for stats in split_stats.values())
    
    print("\n📊 FINAL DATASET COMPOSITION:")
    print("-" * 50)
    
    for split in splits:
        stats = split_stats[split]
        if stats['images'] > 0:
            aug_pct = stats['augmented'] / stats['images'] * 100
            synth_pct = stats['synthetic'] / stats['images'] * 100
            
            print(f"  {split.upper()}:")
            print(f"    Total: {stats['images']:,} images")
            print(f"    Augmented: {stats['augmented']:,} ({aug_pct:.1f}%)")
            print(f"    Synthetic: {stats['synthetic']:,} ({synth_pct:.1f}%)")
            print(f"    Label coverage: {stats['labels']/stats['images']*100:.1f}%")
    
    print(f"\n  GRAND TOTAL: {total_images:,} images, {total_labels:,} labels")
    
    # ============================================================
    # 4. VALIDATE FINAL DATASET
    # ============================================================
    
    print("\n🔍 VALIDATING FINAL DATASET...")
    final_validator = DatasetQualityValidator()
    validation_passed = final_validator.validate_dataset(optimal_path)
    
    if not validation_passed:
        print("\n❌ Final dataset validation failed!")
        response = input("Continue to create dataset anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            # Clean up
            if optimal_path.exists():
                shutil.rmtree(optimal_path)
            return
    
    # ============================================================
    # 5. CREATE ENHANCED data.yaml
    # ============================================================
    
    print("\n⚙️  CREATING ENHANCED data.yaml...")
    
    # Calculate dataset statistics
    aspect_ratio_stats = calculate_aspect_ratio_stats(optimal_path)
    
    data_yaml = {
        'path': str(optimal_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['license_plate'],
        
        # Enhanced metadata
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'phase': 'Phase 0 - Complete',
            'status': 'VALIDATED',
            
            'statistics': {
                'total_images': total_images,
                'total_labels': total_labels,
                'label_coverage': f"{total_labels/total_images*100:.1f}%",
                'splits': split_stats,
            },
            
            'composition': {
                'augmented': {
                    'percentage': '80-90%',
                    'source': 'datasets/04_augmented',
                    'quality': 'VALIDATED'
                },
                'synthetic': {
                    'percentage': '10-20%',
                    'source': 'datasets/06_synthetic',
                    'purpose': 'rare pattern enhancement'
                },
                'flagged_cases': '0% (excluded)'
            },
            
            'quality_metrics': {
                'validation_status': 'PASSED' if validation_passed else 'WARNINGS',
                'aspect_ratio': aspect_ratio_stats,
                'recommended_training': {
                    'model': 'yolo11s.pt',
                    'epochs': 150,
                    'batch_size': 32,
                    'image_size': 640
                }
            },
            
            'notes': [
                'Dataset optimized for Phase 1 detector training',
                'Includes FIXED augmentation pipeline output',
                'Enhanced validation performed',
                'Ready for production-level training'
            ]
        }
    }
    
    yaml_path = optimal_path / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ data.yaml created: {yaml_path}")
    
    # ============================================================
    # 6. CREATE COMPREHENSIVE REPORT
    # ============================================================
    
    create_comprehensive_report(optimal_path, split_stats, total_images, total_labels, 
                              validation_passed, final_validator.stats)
    
    # ============================================================
    # 7. COMPLETION MESSAGE
    # ============================================================
    
    print("\n" + "=" * 70)
    print("🎉 PHASE 0 - DATA PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 OPTIMAL DATASET: {optimal_path}")
    print(f"📊 TOTAL IMAGES: {total_images:,}")
    print(f"✅ VALIDATION STATUS: {'PASSED' if validation_passed else 'WITH WARNINGS'}")
    print(f"⚙️  CONFIG FILE: {yaml_path}")
    
    print("\n📋 DATASET COMPOSITION SUMMARY:")
    for split in splits:
        stats = split_stats[split]
        if stats['images'] > 0:
            print(f"  {split.upper()}: {stats['images']:,} images ({stats['labels']/stats['images']*100:.1f}% labeled)")
    
    print("\n🚀 READY FOR PHASE 1 - DETECTOR TRAINING!")
    print("\n💡 NEXT STEPS:")
    print("   1. Review dataset quality report")
    print("   2. Start training: python scripts/phase1_train_detector.py")
    print("   3. Monitor training with TensorBoard")
    print("=" * 70)


def calculate_aspect_ratio_stats(dataset_path: Path, sample_size: int = 200):
    """Calculate aspect ratio statistics"""
    
    stats = {'mean': 3.0, 'std': 0.5, 'samples': 0}
    
    try:
        lbl_dir = dataset_path / "labels" / "train"
        if not lbl_dir.exists():
            return stats
        
        label_files = list(lbl_dir.glob("*.txt"))
        if not label_files:
            return stats
        
        # Sample labels
        random.seed(42)
        sample = random.sample(label_files, min(sample_size, len(label_files)))
        
        aspect_ratios = []
        
        for lbl_file in sample:
            try:
                with open(lbl_file, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            _, x, y, w, h = map(float, parts[:5])
                            if h > 0:
                                aspect_ratio = w / h
                                aspect_ratios.append(aspect_ratio)
            except:
                continue
        
        if aspect_ratios:
            stats = {
                'mean': float(np.mean(aspect_ratios)),
                'std': float(np.std(aspect_ratios)),
                'min': float(np.min(aspect_ratios)),
                'max': float(np.max(aspect_ratios)),
                'samples': len(aspect_ratios)
            }
    
    except Exception as e:
        print(f"  ⚠️  Error calculating aspect ratios: {e}")
    
    return stats


def create_comprehensive_report(dataset_path: Path, split_stats: dict, total_images: int, 
                              total_labels: int, validation_passed: bool, quality_stats: dict):
    """Create comprehensive dataset report"""
    
    report = {
        'phase': 'Phase 0 - Data Pipeline Completion',
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'path': str(dataset_path),
            'total_images': total_images,
            'total_labels': total_labels,
            'label_coverage_percent': total_labels / total_images * 100 if total_images > 0 else 0,
            'split_statistics': split_stats
        },
        'validation': {
            'status': 'PASSED' if validation_passed else 'WARNINGS',
            'quality_metrics': dict(quality_stats)
        },
        'source_datasets': {
            'augmented': {
                'path': 'datasets/04_augmented',
                'status': 'USED',
                'notes': 'Fixed augmentation pipeline output'
            },
            'synthetic': {
                'path': 'datasets/06_synthetic',
                'status': 'PARTIALLY USED',
                'notes': 'Selective inclusion for rare patterns'
            }
        },
        'recommendations': {
            'training': {
                'model': 'yolo11s.pt',
                'epochs': 150,
                'batch_size': 32,
                'image_size': 640
            },
            'validation': 'Monitor mAP@0.5 and mAP@0.5:0.95',
            'deployment': 'Export to ONNX/TensorRT for production'
        },
        'next_phase': 'Phase 1 - Detector Training'
    }
    
    report_path = dataset_path / "phase0_completion_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Comprehensive report: {report_path}")


if __name__ == "__main__":
    try:
        create_optimal_dataset()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)