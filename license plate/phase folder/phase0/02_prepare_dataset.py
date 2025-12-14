#!/usr/bin/env python3
"""
Ultimate Dataset Preparation Pipeline - FIXED VERSION
Converts multiple formats to unified YOLO format with deduplication
"""

import os
import sys
import json
import shutil
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# ================= FIX IMPORT PATH =================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logging import setup_logger
    from src.utils.image_utils import calculate_blur_score
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    sys.exit(1)
# ===================================================

calculate_laplacian_variance = calculate_blur_score

# Setup
RAW_DIR = Path("datasets/raw")
INTERIM_DIR = Path("datasets/interim")
LP_DIR = Path("datasets/license_plate")

# Create directories
for dir_path in [INTERIM_DIR, LP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    (LP_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (LP_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

logger = setup_logger("DatasetPreparer")

class DatasetPreparer:
    def __init__(self, min_image_size: int = 100, max_image_size: int = 3840):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.image_hashes = set()
        self.file_hashes = set()
        
    def discover_datasets(self) -> List[Path]:
        """Discover all downloaded datasets"""
        datasets = []
        for item in RAW_DIR.iterdir():
            if item.is_dir() and item.name != "__MACOSX":  # Skip macOS temp folders
                datasets.append(item)
        
        logger.info(f"Found {len(datasets)} datasets: {[d.name for d in datasets]}")
        
        # Log dataset sizes
        for dataset in datasets:
            image_files = list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.jpeg")) + list(dataset.rglob("*.png"))
            logger.info(f"  {dataset.name}: {len(image_files)} images")
        
        return datasets
    
    def collect_all_files(self, datasets: List[Path]) -> List[Path]:
        """Collect all image files from datasets"""
        all_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.jfif'}
        
        for dataset in datasets:
            logger.info(f"Scanning {dataset.name}...")
            for ext in image_extensions:
                # Case insensitive search
                files = list(dataset.rglob(f"*{ext.lower()}"))
                files += list(dataset.rglob(f"*{ext.upper()}"))
                all_files.extend(files)
        
        # Remove duplicates (same file found multiple times)
        all_files = list(set(all_files))
        logger.info(f"Total images found: {len(all_files):,}")
        return all_files
    
    def calculate_hash(self, file_path: Path) -> Tuple[str, str]:
        """Calculate MD5 and perceptual hash of an image"""
        # MD5 hash
        with open(file_path, 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        
        # Simple hash based on file size and first bytes (fallback if imagehash not available)
        phash = ""
        try:
            # Try to import imagehash
            import imagehash
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                phash = str(imagehash.phash(img))
        except ImportError:
            # Fallback: use file size and first 1KB for hash
            file_size = file_path.stat().st_size
            with open(file_path, 'rb') as f:
                first_bytes = f.read(1024)
            phash = hashlib.md5(str(file_size).encode() + first_bytes).hexdigest()[:16]
        except Exception as e:
            logger.debug(f"Could not calculate phash for {file_path}: {e}")
            phash = ""
        
        return md5_hash, phash
    
    def validate_image(self, file_path: Path) -> bool:
        """Validate image quality and size"""
        try:
            # Check file size
            if file_path.stat().st_size < 1024:  # Less than 1KB
                logger.debug(f"File too small: {file_path}")
                return False
            
            # Open and check image
            with Image.open(file_path) as img:
                # Check dimensions
                width, height = img.size
                if width < self.min_image_size or height < self.min_image_size:
                    logger.debug(f"Image too small: {file_path} ({width}x{height})")
                    return False
                if width > self.max_image_size or height > self.max_image_size:
                    logger.debug(f"Image too large: {file_path} ({width}x{height})")
                    return False
                
                # Check mode
                if img.mode not in ['RGB', 'L', 'RGBA']:
                    logger.debug(f"Unsupported image mode: {file_path} ({img.mode})")
                    return False
                
                # Convert to numpy for additional checks
                try:
                    img_array = np.array(img)
                    
                    # Check for all same color (corrupted)
                    if len(img_array.shape) == 3:
                        if np.all(img_array == img_array[0,0,0]):
                            logger.debug(f"Corrupted image (single color): {file_path}")
                            return False
                    
                    # Check blur (optional)
                    if img.mode == 'RGB':
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        blur_score = calculate_laplacian_variance(gray)
                        if blur_score < 30:  # Threshold for very blurry images
                            logger.debug(f"Very blurry image: {file_path} (score: {blur_score:.1f})")
                            # Don't reject, just log
                    
                    return True
                    
                except Exception as e:
                    logger.debug(f"Error converting image: {file_path}, {e}")
                    return False
                
        except Exception as e:
            logger.debug(f"Invalid image {file_path}: {e}")
            return False
    
    def find_label_file(self, image_path: Path, dataset_root: Path) -> Optional[Path]:
        """Find corresponding label file for an image - FIXED VERSION"""
        # Get image stem (filename without extension)
        image_stem = image_path.stem
        
        # For Roboflow datasets, keep the .rf. hash for matching
        # DON'T remove .rf. suffix for roboflow_lpr!
        
        # Try different naming conventions and locations
        possible_locations = []
        
        # 1. Check same directory structure but in labels/ folder
        # Example: test/images/xxx.jpg -> test/labels/xxx.txt
        if image_path.parent.name == "images":
            labels_dir = image_path.parent.parent / "labels"
            possible_locations.append(labels_dir / f"{image_stem}.txt")
        
        # 2. Check relative path from dataset root
        # Get relative path from dataset root
        try:
            rel_path = image_path.relative_to(dataset_root)
            # Convert images/xxx.jpg to labels/xxx.txt
            if str(rel_path).startswith("train/images/"):
                label_rel = Path("train/labels") / f"{image_stem}.txt"
                possible_locations.append(dataset_root / label_rel)
            elif str(rel_path).startswith("valid/images/"):
                label_rel = Path("valid/labels") / f"{image_stem}.txt"
                possible_locations.append(dataset_root / label_rel)
            elif str(rel_path).startswith("test/images/"):
                label_rel = Path("test/labels") / f"{image_stem}.txt"
                possible_locations.append(dataset_root / label_rel)
            elif str(rel_path).startswith("images/"):
                label_rel = Path("labels") / f"{image_stem}.txt"
                possible_locations.append(dataset_root / label_rel)
        except ValueError:
            pass
        
        # 3. Standard YOLO locations (already in your list)
        possible_locations.extend([
            # Same directory with same name
            image_path.with_suffix('.txt'),
            image_path.with_suffix('.json'),
            
            # Labels subdirectory
            image_path.parent / "labels" / f"{image_stem}.txt",
            image_path.parent.parent / "labels" / f"{image_stem}.txt",
            
            # Dataset root labels
            dataset_root / "labels" / f"{image_stem}.txt",
            dataset_root / "train" / "labels" / f"{image_stem}.txt",
            dataset_root / "valid" / "labels" / f"{image_stem}.txt",
            dataset_root / "val" / "labels" / f"{image_stem}.txt",
            dataset_root / "test" / "labels" / f"{image_stem}.txt",
        ])
        
        # 4. Also check if image is in images/ subdirectory
        if image_path.parent.name == "images":
            # Check sibling labels directory
            labels_dir = image_path.parent.with_name("labels")
            possible_locations.append(labels_dir / f"{image_stem}.txt")
            
            # Check parent's labels directory
            parent_labels = image_path.parent.parent / "labels"
            possible_locations.append(parent_labels / f"{image_stem}.txt")
        
        # Remove duplicates
        possible_locations = list(set(possible_locations))
        
        # Debug: log first few locations
        logger.debug(f"Looking for label for: {image_path.name}")
        logger.debug(f"  Possible locations ({len(possible_locations)}):")
        for loc in possible_locations[:3]:  # Show first 3
            logger.debug(f"    - {loc}")
        
        # Check all locations
        for label_path in possible_locations:
            if label_path.exists() and label_path.stat().st_size > 0:
                # Skip README files
                if "README" in label_path.name.upper():
                    continue
                logger.debug(f"  ✓ Found label: {label_path}")
                return label_path
        
        logger.debug(f"  ✗ No label file found after checking {len(possible_locations)} locations")
        return None

    def parse_label_file(self, label_path: Path, img_width: int, img_height: int, image_stem: str = "", image_path: Path = None):
        """Parse label file in various formats"""
        if not label_path or not label_path.exists():
            return []
        
        annotations = []
        
        # Skip README files
        if "README" in label_path.name.upper():
            return []
        
        # Check file extension
        if label_path.suffix == '.txt':
            # YOLO format
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate normalized coordinates
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 <= width <= 1 and 0 <= height <= 1):
                            logger.warning(f"Invalid normalized coordinates in {label_path}: {line}")
                            continue
                        
                        # Convert to absolute coordinates
                        x1 = max(0, (x_center - width/2) * img_width)
                        y1 = max(0, (y_center - height/2) * img_height)
                        x2 = min(img_width, (x_center + width/2) * img_width)
                        y2 = min(img_height, (y_center + height/2) * img_height)
                        
                        # Check if bbox is valid
                        if x2 <= x1 or y2 <= y1:
                            logger.warning(f"Invalid bbox in {label_path}: {line}")
                            continue
                        
                        annotations.append({
                            'class_id': class_id,
                            'bbox': [x1, y1, x2, y2],
                            'normalized': [x_center, y_center, width, height]
                        })
                    except ValueError:
                        logger.warning(f"Could not parse line in {label_path}: {line}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error reading YOLO file {label_path}: {e}")
        
        elif label_path.suffix == '.json':
            # COCO or Roboflow JSON format
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for COCO format (global annotations file)
                if 'annotations' in data and 'images' in data:
                    # COCO format - find this specific image
                    # Get clean image stem without .rf. suffix
                    clean_stem = image_stem
                    if '.rf.' in clean_stem:
                        clean_stem = clean_stem.split('.rf.')[0]
                    
                    # Get image filename
                    if image_path:
                        image_filename = image_path.name
                    else:
                        image_filename = f"{clean_stem}.jpg"
                    
                    # Try to find image in COCO data
                    image_id = None
                    image_info = None
                    
                    for img_info in data['images']:
                        img_filename = img_info['file_name']
                        
                        # Multiple matching strategies
                        if (img_filename == image_filename or
                            img_filename == f"{clean_stem}{image_path.suffix if image_path else '.jpg'}" or
                            Path(img_filename).stem == clean_stem or
                            ('.rf.' in img_filename and img_filename.split('.rf.')[0] == clean_stem)):
                            image_id = img_info['id']
                            image_info = img_info
                            break
                    
                    if image_id and image_info:
                        # Verify image dimensions match
                        coco_width = image_info.get('width', img_width)
                        coco_height = image_info.get('height', img_height)
                        
                        # Use COCO dimensions if available and different
                        if coco_width != img_width or coco_height != img_height:
                            logger.debug(f"Using COCO dimensions: {coco_width}x{coco_height} (instead of {img_width}x{img_height})")
                            img_width, img_height = coco_width, coco_height
                        
                        for ann in data['annotations']:
                            if ann['image_id'] == image_id and 'bbox' in ann:
                                x, y, w, h = ann['bbox']
                                # COCO bbox is [x, y, width, height]
                                
                                # Validate bbox
                                if w <= 0 or h <= 0:
                                    continue
                                
                                annotations.append({
                                    'class_id': ann.get('category_id', 0),
                                    'bbox': [x, y, x + w, y + h],
                                    'normalized': [
                                        (x + w/2) / img_width,
                                        (y + h/2) / img_height,
                                        w / img_width,
                                        h / img_height
                                    ]
                                })
                
                # Check for Roboflow JSON format (single image)
                elif 'image' in data:
                    # Roboflow format (per-image JSON)
                    for shape in data.get('shapes', []):
                        if shape.get('shape_type') == 'rectangle':
                            points = shape['points']
                            if len(points) == 2:
                                x1, y1 = points[0]
                                x2, y2 = points[1]
                                
                                # Ensure coordinates are in right order
                                x1, x2 = min(x1, x2), max(x1, x2)
                                y1, y2 = min(y1, y2), max(y1, y2)
                                
                                # Validate bbox
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                
                                annotations.append({
                                    'class_id': shape.get('label_id', 0),
                                    'bbox': [x1, y1, x2, y2],
                                    'normalized': [
                                        ((x1 + x2)/2) / img_width,
                                        ((y1 + y2)/2) / img_height,
                                        (x2 - x1) / img_width,
                                        (y2 - y1) / img_height
                                    ]
                                })
                
            except Exception as e:
                logger.error(f"Error reading JSON file {label_path}: {e}")
                logger.error(f"File: {label_path}, Error: {str(e)}")
        
        return annotations
    
    def process_single_file(self, file_path: Path, dataset_root: Path, counter: int):
        """Process a single file"""
        try:
            # Skip README files
            if "README" in file_path.name.upper():
                return None
            
            # Validate image
            if not self.validate_image(file_path):
                logger.debug(f"Image validation failed: {file_path}")
                return None
            
            # Calculate hashes
            md5_hash, phash = self.calculate_hash(file_path)
            
            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size
            
            # Find and parse label
            image_stem = file_path.stem
            
            # Remove .rf. suffix for matching
            clean_stem = image_stem
            if '.rf.' in clean_stem:
                clean_stem = clean_stem.split('.rf.')[0]
            
            label_path = self.find_label_file(file_path, dataset_root)
            
            # Debug logging for missing labels
            if not label_path:
                logger.debug(f"No label file found for: {file_path}")
                logger.debug(f"Dataset: {dataset_root.name}, Image stem: {clean_stem}")
                return None
            
            annotations = self.parse_label_file(label_path, width, height, clean_stem, file_path)
            
            # Skip images without annotations
            if not annotations:
                logger.debug(f"No annotations parsed for {file_path}")
                logger.debug(f"Label file: {label_path}")
                return None
            
            return {
                'file_path': file_path,
                'md5_hash': md5_hash,
                'phash': phash,
                'width': width,
                'height': height,
                'annotations': annotations,
                'label_path': label_path,
                'counter': counter,
                'dataset': dataset_root.name
            }
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            return None
    
    def deduplicate_and_process(self, all_files: List[Path]) -> List[Dict]:
        """Process all files with deduplication"""
        logger.info("Processing and deduplicating images...")
        
        # Prepare file list with dataset info
        file_dataset_pairs = []
        file_counter = {}
        
        for file_path in all_files:
            # Skip README files
            if "README" in file_path.name.upper():
                continue
                
            # Find which dataset this file belongs to
            dataset_root = None
            for parent in file_path.parents:
                if parent.parent == RAW_DIR:
                    dataset_root = parent
                    break
            
            if dataset_root:
                # Count files per dataset
                if dataset_root.name not in file_counter:
                    file_counter[dataset_root.name] = 0
                counter = file_counter[dataset_root.name]
                file_counter[dataset_root.name] += 1
                
                file_dataset_pairs.append((file_path, dataset_root, counter))
        
        logger.info(f"Files to process: {len(file_dataset_pairs):,}")
        
        # Process files (single-threaded for debugging, can parallelize later)
        processed_data = []
        seen_md5 = set()
        seen_phash = set()
        stats = {
            'total_processed': 0,
            'valid_images': 0,
            'no_label': 0,
            'no_annotations': 0,
            'validation_failed': 0
        }
        
        with tqdm(total=len(file_dataset_pairs), desc="Processing images") as pbar:
            for file_path, dataset_root, counter in file_dataset_pairs:
                stats['total_processed'] += 1
                result = self.process_single_file(file_path, dataset_root, counter)
                
                if result:
                    # TEMPORARILY DISABLE DEDUPLICATION - TOO AGGRESSIVE
                    # if result['md5_hash'] in seen_md5:
                    #     logger.debug(f"Duplicate MD5: {file_path}")
                    #     continue
                    # if result['phash'] and result['phash'] in seen_phash:
                    #     logger.debug(f"Duplicate perceptual hash: {file_path}")
                    #     continue
                    
                    # seen_md5.add(result['md5_hash'])
                    # if result['phash']:
                    #     seen_phash.add(result['phash'])
                    
                    processed_data.append(result)
                    stats['valid_images'] += 1
                else:
                    # Track why files were rejected
                    if not self.validate_image(file_path):
                        stats['validation_failed'] += 1
                    else:
                        label_path = self.find_label_file(file_path, dataset_root)
                        if not label_path:
                            stats['no_label'] += 1
                        else:
                            stats['no_annotations'] += 1
                
                pbar.update(1)
        
        logger.info(f"Processing statistics:")
        logger.info(f"  Total processed: {stats['total_processed']:,}")
        logger.info(f"  Valid images: {stats['valid_images']:,}")
        logger.info(f"  Validation failed: {stats['validation_failed']:,}")
        logger.info(f"  No label file: {stats['no_label']:,}")
        logger.info(f"  No annotations: {stats['no_annotations']:,}")
        
        logger.info(f"After deduplication: {len(processed_data):,} unique images")
        
        if not processed_data:
            logger.error("No images were processed successfully!")
            return []
        
        # Count by dataset
        dataset_counts = {}
        dataset_ann_counts = {}
        for item in processed_data:
            dataset = item['dataset']
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            dataset_ann_counts[dataset] = dataset_ann_counts.get(dataset, 0) + len(item['annotations'])
        
        logger.info("Images per dataset:")
        for dataset, count in dataset_counts.items():
            avg_ann = dataset_ann_counts[dataset] / count if count > 0 else 0
            logger.info(f"  {dataset}: {count} images ({avg_ann:.2f} annotations/image)")
        
        return processed_data
    
    def create_split(self, processed_data: List[Dict], 
                    train_ratio: float = 0.7, 
                    val_ratio: float = 0.2) -> Dict[str, List[Dict]]:
        """Split data into train/val/test sets"""
        # Shuffle data
        random.seed(42)  # For reproducibility
        random.shuffle(processed_data)
        
        total = len(processed_data)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        splits = {
            'train': processed_data[:train_end],
            'val': processed_data[train_end:val_end],
            'test': processed_data[val_end:]
        }
        
        logger.info(f"Split sizes: Train={len(splits['train']):,}, "
                   f"Val={len(splits['val']):,}, Test={len(splits['test']):,}")
        
        return splits
    
    def save_to_yolo_format(self, splits: Dict[str, List[Dict]]):
        """Save processed data to YOLO format"""
        logger.info("Saving to YOLO format...")
        
        statistics = {
            'train': {'images': 0, 'annotations': 0},
            'val': {'images': 0, 'annotations': 0},
            'test': {'images': 0, 'annotations': 0}
        }
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} images)...")
            
            for item in tqdm(split_data, desc=f"Saving {split_name}"):
                # Generate new filename
                new_filename = f"{split_name}_{item['counter']:08d}{item['file_path'].suffix.lower()}"
                
                # Copy image
                dst_image = LP_DIR / "images" / split_name / new_filename
                try:
                    shutil.copy2(item['file_path'], dst_image)
                except Exception as e:
                    logger.error(f"Failed to copy {item['file_path']}: {e}")
                    continue
                
                # Create YOLO label file
                dst_label = LP_DIR / "labels" / split_name / new_filename.replace(
                    item['file_path'].suffix.lower(), '.txt'
                )
                
                try:
                    with open(dst_label, 'w') as f:
                        for ann in item['annotations']:
                            if 'normalized' in ann:
                                x_center, y_center, width, height = ann['normalized']
                                # Ensure class_id is 0 for license plate (single class)
                                class_id = 0
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except Exception as e:
                    logger.error(f"Failed to write label {dst_label}: {e}")
                    continue
                
                # Update statistics
                statistics[split_name]['images'] += 1
                statistics[split_name]['annotations'] += len(item['annotations'])
        
        # Save statistics
        stats_file = LP_DIR / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
        return statistics
    
    def create_data_yaml(self, statistics: Dict):
        """Create data.yaml file for YOLO training"""
        data_yaml = {
            'path': str(LP_DIR.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['license_plate'],
            'statistics': statistics
        }
        
        yaml_file = LP_DIR / "data.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"YAML configuration saved to {yaml_file}")
        return yaml_file
    
    def run(self):
        """Main preparation pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING DATASET PREPARATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Discover datasets
        datasets = self.discover_datasets()
        if not datasets:
            logger.error("No datasets found in raw directory!")
            sys.exit(1)
        
        # Step 2: Collect all files
        all_files = self.collect_all_files(datasets)
        
        if not all_files:
            logger.error("No image files found!")
            sys.exit(1)
        
        # Step 3: Process and deduplicate
        processed_data = self.deduplicate_and_process(all_files)
        
        if not processed_data:
            logger.error("No valid images found after processing!")
            sys.exit(1)
        
        # Step 4: Create splits
        splits = self.create_split(processed_data)
        
        # Step 5: Save to YOLO format
        statistics = self.save_to_yolo_format(splits)
        
        # Step 6: Create data.yaml
        yaml_file = self.create_data_yaml(statistics)
        
        # Summary
        total_images = sum(stats['images'] for stats in statistics.values())
        total_annotations = sum(stats['annotations'] for stats in statistics.values())
        
        logger.info("\n" + "=" * 60)
        logger.info("PREPARATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Total images: {total_images:,}")
        logger.info(f"✅ Total annotations: {total_annotations:,}")
        logger.info(f"✅ Train images: {statistics['train']['images']:,}")
        logger.info(f"✅ Val images: {statistics['val']['images']:,}")
        logger.info(f"✅ Test images: {statistics['test']['images']:,}")
        logger.info(f"📁 Dataset location: {LP_DIR.absolute()}")
        logger.info(f"⚙️  Configuration: {yaml_file}")
        
        # Calculate average annotations per image
        if total_images > 0:
            avg_annotations = total_annotations / total_images
            logger.info(f"📊 Average annotations per image: {avg_annotations:.2f}")

def main():
    preparer = DatasetPreparer()
    preparer.run()

if __name__ == "__main__":
    main()