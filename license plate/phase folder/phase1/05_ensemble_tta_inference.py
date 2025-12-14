"""
ENSEMBLE + TTA INFERENCE FOR MAXIMUM ACCURACY
Senior-level inference with ensemble and test-time augmentation
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import List, Dict, Tuple, Any
import cv2
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.detector.ultimate.ensemble_tta import EnsembleTTAInference
from ultralytics import YOLO

class UltimateEnsembleInference:
    """Ultimate inference with ensemble and TTA"""
    
    def __init__(self, model_paths: List[str], device='cuda'):
        self.model_paths = model_paths
        self.device = device
        
        # Load all models
        self.models = []
        for path in model_paths:
            print(f"📦 Loading model: {Path(path).name}")
            model = YOLO(path)
            model.to(device)
            self.models.append(model)
        
        # Initialize TTA strategies
        self.tta_strategies = [
            'none',           # Original
            'flip_h',         # Horizontal flip
            'flip_v',         # Vertical flip
            'rotate_90',      # 90 degree rotation
            'rotate_180',     # 180 degree rotation
            'rotate_270',     # 270 degree rotation
            'scale_0.8',      # Scale down
            'scale_1.2',      # Scale up
        ]
        
        print(f"🚀 Ensemble Inference Initialized")
        print(f"📊 Number of models: {len(self.models)}")
        print(f"🎯 TTA strategies: {len(self.tta_strategies)}")
        print(f"💻 Device: {device}")
    
    def apply_tta(self, image: np.ndarray, strategy: str) -> np.ndarray:
        """Apply test-time augmentation"""
        if strategy == 'none':
            return image
        elif strategy == 'flip_h':
            return cv2.flip(image, 1)
        elif strategy == 'flip_v':
            return cv2.flip(image, 0)
        elif strategy == 'rotate_90':
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif strategy == 'rotate_180':
            return cv2.rotate(image, cv2.ROTATE_180)
        elif strategy == 'rotate_270':
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif strategy == 'scale_0.8':
            h, w = image.shape[:2]
            new_h, new_w = int(h * 0.8), int(w * 0.8)
            return cv2.resize(image, (new_w, new_h))
        elif strategy == 'scale_1.2':
            h, w = image.shape[:2]
            new_h, new_w = int(h * 1.2), int(w * 1.2)
            return cv2.resize(image, (new_w, new_h))
        else:
            return image
    
    def inference_with_tta(self, image_path: str, 
                          confidence_threshold: float = 0.25,
                          iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Run inference with all TTA strategies and ensemble"""
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        all_predictions = []
        
        # Run each model with each TTA strategy
        for model_idx, model in enumerate(self.models):
            model_predictions = []
            
            for strategy in self.tta_strategies:
                # Apply TTA
                tta_image = self.apply_tta(image, strategy)
                
                # Run inference
                results = model(tta_image, 
                              conf=confidence_threshold,
                              iou=iou_threshold,
                              verbose=False)
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    # Apply inverse TTA to boxes
                    inv_boxes = self.apply_inverse_tta(boxes, strategy, image.shape)
                    
                    model_predictions.append({
                        'strategy': strategy,
                        'boxes': inv_boxes,
                        'confidences': confs,
                        'classes': classes
                    })
            
            all_predictions.append(model_predictions)
        
        # Ensemble predictions
        ensemble_results = self.ensemble_predictions(all_predictions)
        
        return ensemble_results
    
    def apply_inverse_tta(self, boxes: np.ndarray, strategy: str, 
                         original_shape: Tuple[int, int]) -> np.ndarray:
        """Apply inverse TTA to transform boxes back to original space"""
        h, w = original_shape[:2]
        
        if strategy == 'none':
            return boxes
        elif strategy == 'flip_h':
            # Flip horizontally: x' = w - x
            inv_boxes = boxes.copy()
            inv_boxes[:, [0, 2]] = w - inv_boxes[:, [2, 0]]
            return inv_boxes
        elif strategy == 'flip_v':
            # Flip vertically: y' = h - y
            inv_boxes = boxes.copy()
            inv_boxes[:, [1, 3]] = h - inv_boxes[:, [3, 1]]
            return inv_boxes
        elif strategy == 'rotate_90':
            # Rotate 90 clockwise: (x, y) -> (y, w-x)
            inv_boxes = boxes.copy()
            inv_boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 2, 3, 0]]
            inv_boxes[:, [0, 2]] = h - inv_boxes[:, [0, 2]]  # Adjust for rotation
            return inv_boxes
        elif strategy == 'rotate_180':
            # Rotate 180: (x, y) -> (w-x, h-y)
            inv_boxes = boxes.copy()
            inv_boxes[:, [0, 1, 2, 3]] = boxes[:, [2, 3, 0, 1]]
            inv_boxes[:, [0, 2]] = w - inv_boxes[:, [0, 2]]
            inv_boxes[:, [1, 3]] = h - inv_boxes[:, [1, 3]]
            return inv_boxes
        elif strategy == 'rotate_270':
            # Rotate 270: (x, y) -> (h-y, x)
            inv_boxes = boxes.copy()
            inv_boxes[:, [0, 1, 2, 3]] = boxes[:, [3, 0, 1, 2]]
            inv_boxes[:, [1, 3]] = w - inv_boxes[:, [1, 3]]
            return inv_boxes
        elif strategy == 'scale_0.8':
            # Scale 0.8: Multiply coordinates by 1.25
            inv_boxes = boxes.copy() * 1.25
            return inv_boxes
        elif strategy == 'scale_1.2':
            # Scale 1.2: Multiply coordinates by 0.833
            inv_boxes = boxes.copy() * 0.833
            return inv_boxes
        else:
            return boxes
    
    def ensemble_predictions(self, all_predictions: List[List[Dict]], 
                           method: str = 'weighted_voting') -> Dict[str, Any]:
        """Ensemble predictions from multiple models and TTA strategies"""
        
        if method == 'weighted_voting':
            return self.weighted_voting_ensemble(all_predictions)
        elif method == 'nms_ensemble':
            return self.nms_based_ensemble(all_predictions)
        else:
            return self.simple_ensemble(all_predictions)
    
    def weighted_voting_ensemble(self, all_predictions: List[List[Dict]]) -> Dict[str, Any]:
        """Weighted voting ensemble based on model confidence"""
        
        # Collect all predictions
        all_boxes = []
        all_scores = []
        all_weights = []
        
        for model_idx, model_preds in enumerate(all_predictions):
            model_weight = 1.0 / (model_idx + 1)  # Weight decreases with model index
            
            for pred in model_preds:
                if len(pred['boxes']) > 0:
                    for box, conf in zip(pred['boxes'], pred['confidences']):
                        all_boxes.append(box)
                        all_scores.append(conf)
                        all_weights.append(model_weight)
        
        if not all_boxes:
            return {'boxes': np.array([]), 'confidences': np.array([]), 'classes': np.array([])}
        
        # Convert to numpy arrays
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        
        # Apply weighted NMS
        from src.detector.ultimate.ensemble_tta import weighted_nms
        final_boxes, final_scores, final_classes = weighted_nms(
            all_boxes, all_scores, all_weights
        )
        
        return {
            'boxes': final_boxes,
            'confidences': final_scores,
            'classes': final_classes,
            'num_original_predictions': len(all_boxes),
            'num_ensemble_predictions': len(final_boxes)
        }
    
    def benchmark_ensemble(self, test_images: List[str], output_dir: str):
        """Benchmark ensemble performance"""
        print(f"\n🏎️  BENCHMARKING ENSEMBLE INFERENCE")
        
        results = {
            'single_model': {'times': [], 'detections': []},
            'ensemble': {'times': [], 'detections': []},
            'ensemble_tta': {'times': [], 'detections': []}
        }
        
        import time
        
        # Benchmark single model (first model)
        print("   🔬 Single model baseline...")
        for img_path in test_images[:10]:  # Test on 10 images
            start = time.time()
            result = self.models[0](img_path, verbose=False)
            end = time.time()
            
            results['single_model']['times'].append(end - start)
            if result[0].boxes is not None:
                results['single_model']['detections'].append(len(result[0].boxes))
            else:
                results['single_model']['detections'].append(0)
        
        # Benchmark ensemble without TTA
        print("   🔬 Ensemble (no TTA)...")
        for img_path in test_images[:10]:
            start = time.time()
            # Simplified ensemble
            ensemble_preds = []
            for model in self.models:
                result = model(img_path, verbose=False)
                if result[0].boxes is not None:
                    ensemble_preds.append({
                        'boxes': result[0].boxes.xyxy.cpu().numpy(),
                        'confidences': result[0].boxes.conf.cpu().numpy()
                    })
            
            end = time.time()
            results['ensemble']['times'].append(end - start)
            # Count average detections
            if ensemble_preds:
                avg_dets = np.mean([len(p['boxes']) for p in ensemble_preds])
                results['ensemble']['detections'].append(avg_dets)
            else:
                results['ensemble']['detections'].append(0)
        
        # Calculate statistics
        summary = {}
        for method, data in results.items():
            if data['times']:
                summary[method] = {
                    'avg_time_ms': np.mean(data['times']) * 1000,
                    'std_time_ms': np.std(data['times']) * 1000,
                    'avg_detections': np.mean(data['detections']),
                    'speedup_vs_single': 1.0  # Will calculate
                }
        
        # Calculate speedup
        if 'single_model' in summary and 'ensemble' in summary:
            summary['ensemble']['speedup_vs_single'] = (
                summary['single_model']['avg_time_ms'] / 
                summary['ensemble']['avg_time_ms']
            )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_path = output_path / "ensemble_benchmark.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results,
                'timestamp': datetime.now().isoformat(),
                'num_models': len(self.models),
                'num_tta_strategies': len(self.tta_strategies)
            }, f, indent=2)
        
        print(f"\n📊 ENSEMBLE BENCHMARK RESULTS")
        print("=" * 50)
        for method, stats in summary.items():
            print(f"{method:<15} {stats['avg_time_ms']:.1f} ms ± {stats['std_time_ms']:.1f} ms")
            if 'speedup_vs_single' in stats:
                print(f"                Speedup: {stats['speedup_vs_single']:.2f}x")
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble + TTA Inference")
    
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="Paths to model checkpoints")
    
    parser.add_argument("--image", type=str,
                       help="Single image for inference")
    
    parser.add_argument("--image-dir", type=str,
                       help="Directory of images for batch inference")
    
    parser.add_argument("--benchmark", action="store_true", default=True,
                       help="Run benchmark comparison")
    
    parser.add_argument("--output-dir", type=str, default="runs/ensemble",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤝 ENSEMBLE + TTA INFERENCE - SENIOR LEVEL")
    print("=" * 70)
    
    # Initialize ensemble inference
    ensemble = UltimateEnsembleInference(args.models)
    
    # Run single image inference if provided
    if args.image:
        print(f"\n🔍 Running inference on: {args.image}")
        results = ensemble.inference_with_tta(args.image)
        
        print(f"\n📊 RESULTS:")
        print(f"   Number of detections: {len(results['boxes'])}")
        if len(results['boxes']) > 0:
            print(f"   Average confidence: {np.mean(results['confidences']):.3f}")
            print(f"   Min confidence: {np.min(results['confidences']):.3f}")
            print(f"   Max confidence: {np.max(results['confidences']):.3f}")
    
    # Run benchmark if requested
    if args.benchmark and args.image_dir:
        import glob
        test_images = glob.glob(str(Path(args.image_dir) / "*.jpg")) + \
                     glob.glob(str(Path(args.image_dir) / "*.png")) + \
                     glob.glob(str(Path(args.image_dir) / "*.jpeg"))
        
        if test_images:
            print(f"\n📈 Found {len(test_images)} test images")
            summary = ensemble.benchmark_ensemble(test_images[:20], args.output_dir)
    
    print(f"\n✅ Ensemble inference complete!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()