#!/usr/bin/env python3
"""
Ultimate Dataset Analysis with Heatmaps, Statistics, and Advanced Visualizations
Senior-Level analysis for license plate dataset quality assessment
"""

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime  # ✅ ADD THIS IMPORT
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger
from src.utils.image_utils import (
    calculate_blur_score, calculate_brightness_contrast,
    calculate_aspect_ratio, estimate_plate_orientation
)
from src.utils.visualization import Visualizer, create_distribution_plot
from src.utils.detection_utils import convert_bbox_format

# Setup
logger = setup_logger("DatasetAnalyzer")

class UltimateDatasetAnalyzer:
    """Ultimate dataset analyzer with comprehensive statistics and visualizations."""
    
    def __init__(self, config_path: str = "configs/data_license_plate.yaml"):
        """Initialize analyzer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.processed_dir = Path(self.config['paths']['processed_data'])
        self.results_dir = Path(self.config['paths']['results']) / "phase0_reports"
        self.vis_dir = Path(self.config['paths']['results']) / "visualizations"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis data storage
        self.plate_data = []  # List of all plate annotations
        self.image_stats = []  # List of image-level statistics
        self.dataset_stats = defaultdict(dict)
        
        # Color palette for visualizations
        self.colors = px.colors.qualitative.Set3
        
        logger.info("Initialized UltimateDatasetAnalyzer")
    
    def analyze_complete(self) -> Dict:
        """Run complete ultimate dataset analysis.
        
        Returns:
            Comprehensive analysis report
        """
        logger.info("=" * 60)
        logger.info("STARTING ULTIMATE DATASET ANALYSIS")
        logger.info("=" * 60)
        
        # Load dataset
        self._load_dataset_structure()
        
        # Run all analysis modules
        analysis_modules = [
            ("basic_statistics", self._analyze_basic_statistics),
            ("image_quality", self._analyze_image_quality),
            ("bounding_box_analysis", self._analyze_bounding_boxes),
            ("plate_characteristics", self._analyze_plate_characteristics),
            ("dataset_distribution", self._analyze_dataset_distribution),
            ("correlation_analysis", self._analyze_correlations),
            ("rare_pattern_detection", self._detect_rare_patterns),
            ("data_leakage_check", self._check_data_leakage),
            ("quality_assessment", self._assess_dataset_quality)
        ]
        
        analysis_report = {}
        
        for module_name, module_func in analysis_modules:
            logger.info(f"\n🔍 Running: {module_name.replace('_', ' ').title()}")
            try:
                result = module_func()
                analysis_report[module_name] = result
                logger.info(f"✅ Completed: {module_name}")
            except Exception as e:
                logger.error(f"❌ Failed {module_name}: {e}")
                analysis_report[module_name] = {"error": str(e)}
        
        # Generate visualizations
        logger.info("\n🎨 Generating visualizations...")
        self._generate_all_visualizations(analysis_report)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(analysis_report)
        analysis_report['overall_quality_score'] = quality_score
        analysis_report['quality_grade'] = self._get_quality_grade(quality_score)
        
        # Save comprehensive report
        self._save_comprehensive_report(analysis_report)
        
        # Print executive summary
        self._print_executive_summary(analysis_report)
        
        return analysis_report
    
    def _load_dataset_structure(self):
        """Load dataset structure and collect file information."""
        logger.info("Loading dataset structure...")
        
        splits = ['train', 'val', 'test']
        self.dataset_info = {}
        
        for split in splits:
            image_dir = self.processed_dir / "images" / split
            label_dir = self.processed_dir / "labels" / split
            
            if image_dir.exists():
                images = list(image_dir.glob("*.jpg")) + \
                        list(image_dir.glob("*.jpeg")) + \
                        list(image_dir.glob("*.png")) + \
                        list(image_dir.glob("*.bmp"))
                
                self.dataset_info[split] = {
                    'images': images,
                    'label_dir': label_dir,
                    'count': len(images)
                }
        
        total_images = sum(info['count'] for info in self.dataset_info.values())
        logger.info(f"📊 Dataset loaded: {total_images:,} total images")
        
        for split, info in self.dataset_info.items():
            logger.info(f"  {split}: {info['count']:,} images ({info['count']/total_images*100:.1f}%)")
    
    def _analyze_basic_statistics(self) -> Dict:
        """Analyze basic dataset statistics."""
        logger.info("Analyzing basic statistics...")
        
        stats = {
            'total_images': 0,
            'images_per_split': {},
            'labels_per_split': {},
            'images_without_labels': 0,
            'average_plates_per_image': 0
        }
        
        total_plates = 0
        total_images = 0
        
        for split, info in self.dataset_info.items():
            image_count = info['count']
            stats['images_per_split'][split] = image_count
            stats['total_images'] += image_count
            
            # Count labels
            label_count = 0
            plates_in_split = 0
            
            if info['label_dir'].exists():
                label_files = list(info['label_dir'].glob("*.txt"))
                label_count = len(label_files)
                
                # Count plates in labels
                for label_file in tqdm(label_files[:1000], desc=f"Sampling {split} labels", leave=False):
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            plates_in_split += len([l for l in lines if l.strip()])
                    except:
                        pass
            
            stats['labels_per_split'][split] = label_count
            
            if image_count > 0:
                plates_in_split = max(plates_in_split, label_count * 0.8)  # Estimate
                total_plates += plates_in_split
                total_images += image_count
        
        if total_images > 0:
            stats['average_plates_per_image'] = total_plates / total_images
            stats['images_without_labels'] = stats['total_images'] - sum(stats['labels_per_split'].values())
        
        return stats
    
    def _analyze_image_quality(self, sample_size: int = 1000) -> Dict:
        """Analyze image quality metrics with sampling."""
        logger.info(f"Analyzing image quality (sampling {sample_size} images)...")
        
        quality_stats = {
            'blur_scores': [],
            'brightness_values': [],
            'contrast_values': [],
            'resolutions': [],
            'orientation_angles': []
        }
        
        # Collect sample images from all splits
        sample_images = []
        for split, info in self.dataset_info.items():
            split_sample = min(len(info['images']), sample_size // 3)
            if split_sample > 0:
                import random
                random.seed(42)
                sample_images.extend(random.sample(info['images'], split_sample))
        
        if len(sample_images) > sample_size:
            sample_images = sample_images[:sample_size]
        
        # Analyze sampled images
        for img_path in tqdm(sample_images, desc="Analyzing image quality"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Basic quality metrics
                blur_score = calculate_blur_score(img)
                brightness, contrast = calculate_brightness_contrast(img)
                
                quality_stats['blur_scores'].append(blur_score)
                quality_stats['brightness_values'].append(brightness)
                quality_stats['contrast_values'].append(contrast)
                
                # Resolution
                h, w = img.shape[:2]
                quality_stats['resolutions'].append((w, h))
                
                # Try to estimate orientation if we have labels
                label_path = self.dataset_info[img_path.parent.name]['label_dir'] / f"{img_path.stem}.txt"
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        if lines:
                            # Use first annotation for orientation estimation
                            parts = lines[0].strip().split()
                            if len(parts) >= 5:
                                yolo_bbox = list(map(float, parts[1:5]))
                                abs_bbox = convert_bbox_format(yolo_bbox, 'yolo', 'xyxy', (w, h))
                                orientation = estimate_plate_orientation(img, abs_bbox)
                                quality_stats['orientation_angles'].append(orientation)
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"Error analyzing {img_path}: {e}")
        
        # Calculate statistics
        results = {}
        
        for metric, values in quality_stats.items():
            if values:
                values_array = np.array(values)
                results[metric] = {
                    'count': len(values),
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array)),
                    'q1': float(np.percentile(values_array, 25)),
                    'q3': float(np.percentile(values_array, 75))
                }
        
        # Add blur quality assessment
        if 'blur_scores' in results:
            blur_threshold = self.config['cleaning']['blur_threshold_laplacian']
            blur_scores = quality_stats['blur_scores']
            percent_blurry = sum(1 for score in blur_scores if score < blur_threshold) / len(blur_scores) * 100
            
            results['blur_quality'] = {
                'threshold': blur_threshold,
                'percent_blurry': float(percent_blurry),
                'assessment': 'Good' if percent_blurry < 10 else 'Needs Attention'
            }
        
        return results
    
    def _analyze_bounding_boxes(self, sample_size: int = 2000) -> Dict:
        """Analyze bounding box characteristics."""
        logger.info(f"Analyzing bounding boxes (sampling {sample_size})...")
        
        bbox_stats = {
            'widths': [],
            'heights': [],
            'areas': [],
            'aspect_ratios': [],
            'positions_x': [],
            'positions_y': [],
            'area_ratios': []
        }
        
        annotations_collected = 0
        
        for split, info in self.dataset_info.items():
            images_sampled = 0
            max_images_per_split = sample_size // 3
            
            for img_path in info['images']:
                if images_sampled >= max_images_per_split:
                    break
                
                label_path = info['label_dir'] / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                yolo_bbox = list(map(float, parts[1:5]))
                                abs_bbox = convert_bbox_format(yolo_bbox, 'yolo', 'xyxy', (w, h))
                                
                                x1, y1, x2, y2 = abs_bbox
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1
                                bbox_area = bbox_width * bbox_height
                                
                                bbox_stats['widths'].append(bbox_width)
                                bbox_stats['heights'].append(bbox_height)
                                bbox_stats['areas'].append(bbox_area)
                                
                                if bbox_height > 0:
                                    aspect_ratio = bbox_width / bbox_height
                                    bbox_stats['aspect_ratios'].append(aspect_ratio)
                                
                                bbox_stats['positions_x'].append((x1 + x2) / 2 / w)
                                bbox_stats['positions_y'].append((y1 + y2) / 2 / h)
                                bbox_stats['area_ratios'].append(bbox_area / (w * h))
                                
                                annotations_collected += 1
                                
                            except:
                                continue
                    
                    images_sampled += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing {img_path}: {e}")
        
        # Calculate statistics
        results = {}
        
        for metric, values in bbox_stats.items():
            if values:
                values_array = np.array(values)
                results[metric] = {
                    'count': len(values),
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array))
                }
        
        results['total_annotations_analyzed'] = annotations_collected
        
        # Check against thresholds
        if 'areas' in results:
            min_area = self.config['cleaning']['min_bbox_area_px']
            max_area = self.config['cleaning'].get('max_bbox_area_px', float('inf'))
            
            areas = bbox_stats['areas']
            too_small = sum(1 for area in areas if area < min_area)
            too_large = sum(1 for area in areas if area > max_area)
            
            results['area_validation'] = {
                'min_threshold': min_area,
                'max_threshold': max_area,
                'percent_too_small': float(too_small / len(areas) * 100) if areas else 0,
                'percent_too_large': float(too_large / len(areas) * 100) if areas else 0,
                'assessment': 'Good' if (too_small / len(areas) * 100 < 5 and too_large / len(areas) * 100 < 5) else 'Needs Attention'
            }
        
        return results
    
    def _analyze_plate_characteristics(self) -> Dict:
        """Analyze license plate specific characteristics."""
        logger.info("Analyzing plate characteristics...")
        
        # This is a placeholder for advanced plate analysis
        # In a real implementation, this would include:
        # - Plate text analysis (if OCR available)
        # - Color distribution analysis
        # - Font style analysis
        # - Region-specific patterns
        
        return {
            'note': 'Advanced plate analysis requires OCR and plate format parsing',
            'recommended_analyses': [
                'Plate text pattern analysis',
                'Color histogram analysis',
                'Regional pattern detection',
                'Font style classification',
                'Special plate type identification'
            ]
        }
    
    def _analyze_dataset_distribution(self) -> Dict:
        """Analyze dataset distribution across splits and characteristics."""
        logger.info("Analyzing dataset distribution...")
        
        distribution = {
            'split_ratios': {},
            'plate_size_distribution': {},
            'aspect_ratio_distribution': {},
            'position_distribution': {}
        }
        
        # Split ratios
        total_images = sum(info['count'] for info in self.dataset_info.values())
        for split, info in self.dataset_info.items():
            if total_images > 0:
                distribution['split_ratios'][split] = {
                    'count': info['count'],
                    'percentage': info['count'] / total_images * 100
                }
        
        return distribution
    
    def _analyze_correlations(self) -> Dict:
        """Analyze correlations between different metrics."""
        logger.info("Analyzing correlations...")
        
        # Collect data for correlation analysis
        correlation_data = []
        
        for split, info in self.dataset_info.items():
            # Sample images for correlation analysis
            sample_size = min(100, len(info['images']))
            import random
            random.seed(42)
            sample_images = random.sample(info['images'], sample_size) if info['images'] else []
            
            for img_path in sample_images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Calculate image metrics
                    blur_score = calculate_blur_score(img)
                    brightness, contrast = calculate_brightness_contrast(img)
                    
                    correlation_data.append({
                        'image_width': w,
                        'image_height': h,
                        'blur_score': blur_score,
                        'brightness': brightness,
                        'contrast': contrast,
                        'split': split
                    })
                    
                except:
                    continue
        
        if not correlation_data:
            return {'note': 'Insufficient data for correlation analysis'}
        
        # Convert to DataFrame for correlation calculation
        df = pd.DataFrame(correlation_data)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            # Get top correlations
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.3:  # Only report meaningful correlations
                        correlations.append({
                            'variables': f"{col1} vs {col2}",
                            'correlation': float(corr_value),
                            'strength': 'Strong' if abs(corr_value) > 0.7 else 
                                       'Moderate' if abs(corr_value) > 0.5 else 'Weak'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'significant_correlations': correlations,
                'data_points': len(correlation_data)
            }
        
        return {'note': 'No significant correlations found'}
    
    def _detect_rare_patterns(self) -> Dict:
        """Detect rare patterns in the dataset."""
        logger.info("Detecting rare patterns...")
        
        # This would require advanced analysis including:
        # - Plate text extraction and pattern analysis
        # - Unusual aspect ratio detection
        # - Rare color combinations
        # - Uncommon plate positions
        
        return {
            'note': 'Rare pattern detection requires OCR and advanced feature extraction',
            'detection_methods': [
                'DBSCAN clustering on plate features',
                'Outlier detection in aspect ratios',
                'Text pattern frequency analysis',
                'Color histogram outlier detection'
            ]
        }
    
    def _check_data_leakage(self) -> Dict:
        """Check for data leakage between splits."""
        logger.info("Checking for data leakage...")
        
        # Simple check for duplicate filenames across splits
        all_filenames = {}
        duplicate_check = {
            'duplicate_filenames': [],
            'total_checks': 0,
            'leakage_detected': False
        }
        
        for split, info in self.dataset_info.items():
            for img_path in info['images']:
                filename = img_path.name
                if filename in all_filenames:
                    all_filenames[filename].append(split)
                    duplicate_check['duplicate_filenames'].append({
                        'filename': filename,
                        'splits': all_filenames[filename]
                    })
                    duplicate_check['leakage_detected'] = True
                else:
                    all_filenames[filename] = [split]
                
                duplicate_check['total_checks'] += 1
        
        return duplicate_check
    
    def _assess_dataset_quality(self) -> Dict:
        """Assess overall dataset quality."""
        logger.info("Assessing dataset quality...")
        
        quality_assessment = {
            'checks': [],
            'passing_checks': 0,
            'total_checks': 0
        }
        
        # Check 1: Sufficient data size
        total_images = sum(info['count'] for info in self.dataset_info.values())
        quality_assessment['total_checks'] += 1
        if total_images >= 1000:
            quality_assessment['checks'].append({
                'check': 'Minimum dataset size',
                'status': 'PASS',
                'details': f'{total_images:,} images (≥ 1,000 required)'
            })
            quality_assessment['passing_checks'] += 1
        else:
            quality_assessment['checks'].append({
                'check': 'Minimum dataset size',
                'status': 'FAIL',
                'details': f'{total_images:,} images (< 1,000 required)'
            })
        
        # Check 2: Balanced splits
        quality_assessment['total_checks'] += 1
        split_ratios = {}
        for split, info in self.dataset_info.items():
            if total_images > 0:
                split_ratios[split] = info['count'] / total_images
        
        # Check if splits are reasonably balanced
        if all(0.05 < ratio < 0.95 for ratio in split_ratios.values()):
            quality_assessment['checks'].append({
                'check': 'Split balance',
                'status': 'PASS',
                'details': f'Split ratios: {split_ratios}'
            })
            quality_assessment['passing_checks'] += 1
        else:
            quality_assessment['checks'].append({
                'check': 'Split balance',
                'status': 'WARNING',
                'details': f'Unbalanced splits: {split_ratios}'
            })
        
        # Check 3: Train set has most data
        quality_assessment['total_checks'] += 1
        if 'train' in split_ratios and split_ratios['train'] > 0.5:
            quality_assessment['checks'].append({
                'check': 'Train set dominance',
                'status': 'PASS',
                'details': f'Train ratio: {split_ratios.get("train", 0):.1%}'
            })
            quality_assessment['passing_checks'] += 1
        else:
            quality_assessment['checks'].append({
                'check': 'Train set dominance',
                'status': 'WARNING',
                'details': f'Train ratio too low: {split_ratios.get("train", 0):.1%}'
            })
        
        quality_assessment['pass_rate'] = quality_assessment['passing_checks'] / quality_assessment['total_checks'] * 100
        
        return quality_assessment
    
    def _calculate_quality_score(self, analysis_report: Dict) -> float:
        """Calculate overall dataset quality score (0-100)."""
        score = 100.0
        deductions = []
        
        # Deduct for basic issues
        basic_stats = analysis_report.get('basic_statistics', {})
        total_images = basic_stats.get('total_images', 0)
        
        if total_images < 1000:
            deduction = min(20, (1000 - total_images) / 1000 * 40)
            score -= deduction
            deductions.append(f"Small dataset ({total_images} images): -{deduction:.1f}")
        
        # Deduct for blurry images
        image_quality = analysis_report.get('image_quality', {})
        blur_quality = image_quality.get('blur_quality', {})
        percent_blurry = blur_quality.get('percent_blurry', 0)
        
        if percent_blurry > 10:
            deduction = min(15, (percent_blurry - 10) / 10 * 10)
            score -= deduction
            deductions.append(f"High blur percentage ({percent_blurry:.1f}%): -{deduction:.1f}")
        
        # Deduct for bbox issues
        bbox_stats = analysis_report.get('bounding_box_analysis', {})
        area_validation = bbox_stats.get('area_validation', {})
        percent_too_small = area_validation.get('percent_too_small', 0)
        percent_too_large = area_validation.get('percent_too_large', 0)
        
        if percent_too_small > 5:
            deduction = min(10, (percent_too_small - 5) / 5 * 5)
            score -= deduction
            deductions.append(f"Many small bboxes ({percent_too_small:.1f}%): -{deduction:.1f}")
        
        if percent_too_large > 5:
            deduction = min(10, (percent_too_large - 5) / 5 * 5)
            score -= deduction
            deductions.append(f"Many large bboxes ({percent_too_large:.1f}%): -{deduction:.1f}")
        
        # Deduct for data leakage
        leakage_check = analysis_report.get('data_leakage_check', {})
        if leakage_check.get('leakage_detected', False):
            score -= 10
            deductions.append(f"Data leakage detected: -10.0")
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        analysis_report['quality_deductions'] = deductions
        analysis_report['raw_quality_score'] = score
        
        return score
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "F"
    
    def _generate_all_visualizations(self, analysis_report: Dict):
        """Generate all visualizations for the analysis report."""
        vis_dir = self.vis_dir / "dataset_analysis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Split distribution
        self._create_split_distribution_plot(analysis_report, vis_dir)
        
        # 2. Image quality metrics
        self._create_quality_metrics_plots(analysis_report, vis_dir)
        
        # 3. Bounding box statistics
        self._create_bbox_statistics_plots(analysis_report, vis_dir)
        
        # 4. Correlation matrix
        self._create_correlation_matrix_plot(analysis_report, vis_dir)
        
        # 5. Quality score dashboard
        self._create_quality_dashboard(analysis_report, vis_dir)
        
        logger.info(f"📊 Visualizations saved to: {vis_dir}")
    
    def _create_split_distribution_plot(self, analysis_report: Dict, output_dir: Path):
        """Create split distribution visualization."""
        try:
            basic_stats = analysis_report.get('basic_statistics', {})
            split_counts = basic_stats.get('images_per_split', {})
            
            if not split_counts:
                return
            
            # Create interactive pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(split_counts.keys()),
                values=list(split_counts.values()),
                hole=0.3,
                textinfo='label+percent+value',
                textposition='inside',
                marker=dict(colors=self.colors),
                hoverinfo='label+percent+value'
            )])
            
            fig.update_layout(
                title={
                    'text': "Dataset Split Distribution",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                showlegend=True,
                annotations=[dict(
                    text=f"Total: {sum(split_counts.values()):,}",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            
            # Save plots
            fig.write_html(str(output_dir / "split_distribution.html"))
            fig.write_image(str(output_dir / "split_distribution.png"), width=800, height=600)
            
        except Exception as e:
            logger.error(f"Error creating split distribution plot: {e}")
    
    def _create_quality_metrics_plots(self, analysis_report: Dict, output_dir: Path):
        """Create image quality metrics visualizations."""
        try:
            quality_stats = analysis_report.get('image_quality', {})
            
            if not quality_stats:
                return
            
            # Create subplots for different metrics
            metrics_to_plot = ['blur_scores', 'brightness_values', 'contrast_values']
            titles = ['Blur Score Distribution', 'Brightness Distribution', 'Contrast Distribution']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=titles,
                horizontal_spacing=0.1
            )
            
            for i, (metric, title) in enumerate(zip(metrics_to_plot, titles), 1):
                if metric in quality_stats:
                    stats_data = quality_stats[metric]
                    
                    if 'mean' in stats_data:
                        # Create histogram with normal distribution overlay
                        mean = stats_data['mean']
                        std = stats_data['std']
                        
                        # Generate synthetic data based on statistics
                        x = np.linspace(mean - 3*std, mean + 3*std, 1000)
                        y = stats.norm.pdf(x, mean, std)
                        
                        # Add histogram (synthetic for visualization)
                        fig.add_trace(
                            go.Histogram(
                                name=metric.replace('_', ' ').title(),
                                xbins=dict(start=mean-3*std, end=mean+3*std, size=std/2),
                                opacity=0.7,
                                marker_color=self.colors[i-1]
                            ),
                            row=1, col=i
                        )
                        
                        # Add normal distribution curve
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y * stats_data['count'] * std,  # Scale to match histogram
                                mode='lines',
                                name='Normal Fit',
                                line=dict(color='red', width=2),
                                showlegend=(i == 1)
                            ),
                            row=1, col=i
                        )
            
            fig.update_layout(
                title_text="Image Quality Metrics Distribution",
                title_x=0.5,
                height=400,
                showlegend=True,
                bargap=0.05
            )
            
            # Update axes
            for i in range(1, 4):
                fig.update_xaxes(title_text=metrics_to_plot[i-1].replace('_', ' '), row=1, col=i)
                fig.update_yaxes(title_text="Count", row=1, col=i)
            
            fig.write_html(str(output_dir / "quality_metrics.html"))
            fig.write_image(str(output_dir / "quality_metrics.png"), width=1200, height=500)
            
        except Exception as e:
            logger.error(f"Error creating quality metrics plots: {e}")
    
    def _create_bbox_statistics_plots(self, analysis_report: Dict, output_dir: Path):
        """Create bounding box statistics visualizations."""
        try:
            bbox_stats = analysis_report.get('bounding_box_analysis', {})
            
            if not bbox_stats:
                return
            
            # Create aspect ratio distribution
            if 'aspect_ratios' in bbox_stats:
                aspect_stats = bbox_stats['aspect_ratios']
                
                # Create histogram
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=[],  # Would be actual data in full implementation
                    name="Aspect Ratios",
                    nbinsx=50,
                    marker_color=self.colors[0],
                    opacity=0.7
                ))
                
                # Add vertical lines for typical license plate aspect ratios
                fig.add_vline(x=2.5, line_dash="dash", line_color="red", 
                            annotation_text="Typical Min", annotation_position="top right")
                fig.add_vline(x=4.0, line_dash="dash", line_color="red",
                            annotation_text="Typical Max", annotation_position="top left")
                
                fig.update_layout(
                    title={
                        'text': "License Plate Aspect Ratio Distribution",
                        'x': 0.5,
                        'font': dict(size=20)
                    },
                    xaxis_title="Aspect Ratio (Width/Height)",
                    yaxis_title="Count",
                    showlegend=False,
                    height=500
                )
                
                fig.write_html(str(output_dir / "aspect_ratio_distribution.html"))
                fig.write_image(str(output_dir / "aspect_ratio_distribution.png"), width=800, height=500)
            
        except Exception as e:
            logger.error(f"Error creating bbox statistics plots: {e}")
    
    def _create_correlation_matrix_plot(self, analysis_report: Dict, output_dir: Path):
        """Create correlation matrix visualization."""
        try:
            correlation_analysis = analysis_report.get('correlation_analysis', {})
            
            if 'correlation_matrix' not in correlation_analysis:
                return
            
            corr_matrix = correlation_analysis['correlation_matrix']
            
            # Convert to DataFrame for plotting
            df_corr = pd.DataFrame(corr_matrix)
            
            # Create heatmap
            fig = px.imshow(
                df_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                title={
                    'text': "Feature Correlation Matrix",
                    'x': 0.5,
                    'font': dict(size=20)
                },
                height=500,
                width=600
            )
            
            fig.write_html(str(output_dir / "correlation_matrix.html"))
            fig.write_image(str(output_dir / "correlation_matrix.png"))
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix plot: {e}")
    
    def _create_quality_dashboard(self, analysis_report: Dict, output_dir: Path):
        """Create quality assessment dashboard."""
        try:
            quality_score = analysis_report.get('overall_quality_score', 0)
            quality_grade = analysis_report.get('quality_grade', 'F')
            
            # Create gauge chart for quality score
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dataset Quality Score", 'font': {'size': 24}},
                delta={'reference': 80, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'red'},
                        {'range': [50, 70], 'color': 'orange'},
                        {'range': [70, 85], 'color': 'yellow'},
                        {'range': [85, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                font={'color': "darkblue", 'family': "Arial"},
                height=400
            )
            
            # Add grade annotation
            fig.add_annotation(
                x=0.5,
                y=0.3,
                text=f"Grade: {quality_grade}",
                showarrow=False,
                font=dict(size=36, color="darkblue")
            )
            
            fig.write_html(str(output_dir / "quality_dashboard.html"))
            fig.write_image(str(output_dir / "quality_dashboard.png"), width=800, height=500)
            
        except Exception as e:
            logger.error(f"Error creating quality dashboard: {e}")
    
    def _save_comprehensive_report(self, analysis_report: Dict):
        """Save comprehensive analysis report."""
        report_path = self.results_dir / "ultimate_analysis_report.json"
        
        # Add metadata
        analysis_report['metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),  # ✅ Now works
            'analyzer_version': '1.0.0',
            'config_used': 'configs/data_license_plate.yaml'
        }
        
        with open(report_path, 'w') as f:
            # Custom JSON serializer for numpy types
            def default_serializer(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(analysis_report, f, indent=2, default=default_serializer)
        
        logger.info(f"📄 Comprehensive report saved to: {report_path}")
    
    def _print_executive_summary(self, analysis_report: Dict):
        """Print executive summary of the analysis."""
        print("\n" + "=" * 70)
        print("DATASET ANALYSIS - EXECUTIVE SUMMARY")
        print("=" * 70)
        
        # Basic statistics
        basic_stats = analysis_report.get('basic_statistics', {})
        total_images = basic_stats.get('total_images', 0)
        
        print(f"\n📊 BASIC STATISTICS")
        print(f"   Total Images: {total_images:,}")
        
        for split, count in basic_stats.get('images_per_split', {}).items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"   {split.upper()}: {count:,} ({percentage:.1f}%)")
        
        # Quality score
        quality_score = analysis_report.get('overall_quality_score', 0)
        quality_grade = analysis_report.get('quality_grade', 'F')
        
        print(f"\n🎯 QUALITY ASSESSMENT")
        print(f"   Overall Score: {quality_score:.1f}/100")
        print(f"   Grade: {quality_grade}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS")
        
        # Blur assessment
        image_quality = analysis_report.get('image_quality', {})
        blur_quality = image_quality.get('blur_quality', {})
        percent_blurry = blur_quality.get('percent_blurry', 0)
        
        if percent_blurry > 10:
            print(f"   ⚠️  High blur percentage: {percent_blurry:.1f}%")
        else:
            print(f"   ✅ Acceptable blur percentage: {percent_blurry:.1f}%")
        
        # Bbox issues
        bbox_stats = analysis_report.get('bounding_box_analysis', {})
        area_validation = bbox_stats.get('area_validation', {})
        percent_too_small = area_validation.get('percent_too_small', 0)
        percent_too_large = area_validation.get('percent_too_large', 0)
        
        if percent_too_small > 5 or percent_too_large > 5:
            print(f"   ⚠️  Bounding box size issues: {percent_too_small:.1f}% too small, {percent_too_large:.1f}% too large")
        else:
            print(f"   ✅ Bounding box sizes are within acceptable range")
        
        # Data leakage
        leakage_check = analysis_report.get('data_leakage_check', {})
        if leakage_check.get('leakage_detected', False):
            print(f"   ❗ Data leakage detected between splits")
        else:
            print(f"   ✅ No data leakage detected")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        
        if quality_score < 70:
            print(f"   1. Consider adding more training data")
            print(f"   2. Review and clean blurry images")
            print(f"   3. Validate bounding box annotations")
        elif quality_score < 85:
            print(f"   1. Dataset is good for initial training")
            print(f"   2. Consider targeted augmentation")
            print(f"   3. Monitor performance on validation set")
        else:
            print(f"   1. Excellent dataset quality")
            print(f"   2. Ready for production-level training")
            print(f"   3. Consider advanced augmentation techniques")
        
        print(f"\n📁 Reports saved to: {self.results_dir}")
        print(f"🎨 Visualizations saved to: {self.vis_dir / 'dataset_analysis'}")
        print("=" * 70)


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ultimate Dataset Analyzer for License Plate Recognition"
    )
    parser.add_argument(
        "--config", 
        default="configs/data_license_plate.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick analysis with smaller samples"
    )
    parser.add_argument(
        "--visualize-only", 
        action="store_true",
        help="Only generate visualizations from existing report"
    )
    
    args = parser.parse_args()
    
    analyzer = UltimateDatasetAnalyzer(args.config)
    
    if args.visualize_only:
        # Load existing report and generate visualizations
        report_path = analyzer.results_dir / "ultimate_analysis_report.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                analysis_report = json.load(f)
            analyzer._generate_all_visualizations(analysis_report)
            print("Visualizations generated from existing report.")
        else:
            print("No existing analysis report found. Running full analysis...")
            analyzer.analyze_complete()
    else:
        analyzer.analyze_complete()


if __name__ == "__main__":
    main()