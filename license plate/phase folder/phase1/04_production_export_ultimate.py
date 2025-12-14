# scripts/phase1/04_production_export_ultimate.py
"""
ULTIMATE PRODUCTION EXPORT FOR RTX 4060
Export to ONNX FP16, INT8, TensorRT with benchmarking
"""

import torch
import onnx
import onnxruntime as ort
import tensorrt as trt
import yaml
import json
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.detector.ultimate.gpu_monitor import RTX4060Monitor
from ultralytics import YOLO

class UltimateProductionExporter:
    """Ultimate model exporter for RTX 4060 production"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_monitor = RTX4060Monitor()
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device).eval()
        
        print(f"📦 Ultimate Production Exporter Initialized")
        print(f"🎯 Model: {model_path}")
        print(f"💻 Device: {self.device}")
    
    def load_model(self, model_path: str):
        """Load YOLO model from checkpoint"""
        if model_path.endswith('.pt'):
            yolo_model = YOLO(model_path)
            return yolo_model.model
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def export_onnx_fp32(self, output_path: str, input_shape: tuple = (1, 3, 640, 640)):
        """Export to ONNX FP32"""
        
        print(f"\n📤 Exporting ONNX FP32...")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=13,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX FP32 exported: {output_path}")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        
        return output_path
    
    def export_onnx_fp16(self, output_path: str, input_shape: tuple = (1, 3, 640, 640)):
        """Export to ONNX FP16"""
        
        print(f"\n📤 Exporting ONNX FP16...")
        
        # Convert model to FP16
        model_fp16 = self.model.half()
        
        # Create FP16 dummy input
        dummy_input = torch.randn(*input_shape).half().to(self.device)
        
        # Export
        torch.onnx.export(
            model_fp16,
            dummy_input,
            output_path,
            opset_version=13,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX FP16 exported: {output_path}")
        
        return output_path
    
    def quantize_int8(self, onnx_path: str, output_path: str, 
                     calibration_data: List[np.ndarray] = None):
        """Quantize model to INT8 (simplified version)"""
        
        print(f"\n🎯 Quantizing to INT8...")
        
        # For production, use TensorRT or onnxruntime quantization
        # This is a simplified version
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Dynamic quantization
            quantized_model = quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QInt8
            )
            
            print(f"✅ INT8 quantization complete: {output_path}")
            return output_path
            
        except ImportError:
            print("⚠️  INT8 quantization requires onnxruntime>=1.8.0")
            print("   Skipping INT8 quantization")
            return None
    
    def benchmark_model(self, model_path: str, input_shape: tuple = (1, 3, 640, 640), 
                       runs: int = 100) -> Dict[str, Any]:
        """Benchmark model inference speed"""
        
        print(f"\n📊 Benchmarking {model_path}...")
        
        results = {
            'model': str(model_path),
            'input_shape': input_shape,
            'runs': runs,
            'latencies_ms': [],
            'throughput_fps': 0.0,
            'average_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0
        }
        
        # Prepare input
        if model_path.endswith('.onnx'):
            # ONNX model
            sess = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            input_name = sess.get_inputs()[0].name
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                sess.run(None, {input_name: input_data})
            
            # Benchmark
            latencies = []
            for _ in range(runs):
                start = time.perf_counter()
                sess.run(None, {input_name: input_data})
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
            
        elif model_path.endswith('.pt'):
            # PyTorch model
            self.model.eval()
            input_data = torch.randn(*input_shape).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(input_data)
                torch.cuda.synchronize()
            
            # Benchmark
            latencies = []
            with torch.no_grad():
                for _ in range(runs):
                    start = time.perf_counter()
                    _ = self.model(input_data)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)
        
        else:
            print(f"⚠️  Unsupported model format for benchmarking: {model_path}")
            return results
        
        # Calculate statistics
        latencies_np = np.array(latencies)
        results['latencies_ms'] = latencies_np.tolist()
        results['average_latency_ms'] = float(np.mean(latencies_np))
        results['throughput_fps'] = 1000.0 / results['average_latency_ms']
        results['p95_latency_ms'] = float(np.percentile(latencies_np, 95))
        results['p99_latency_ms'] = float(np.percentile(latencies_np, 99))
        
        print(f"   Average Latency: {results['average_latency_ms']:.2f} ms")
        print(f"   Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"   P95 Latency: {results['p95_latency_ms']:.2f} ms")
        print(f"   P99 Latency: {results['p99_latency_ms']:.2f} ms")
        
        return results
    
    def compare_formats(self, output_dir: str):
        """Compare performance of different model formats"""
        
        print(f"\n⚖️  COMPARING MODEL FORMATS")
        print("=" * 50)
        
        results = {}
        
        # Benchmark original PyTorch model
        print(f"\n🎯 PyTorch Model:")
        results['pytorch'] = self.benchmark_model(str(self.model_path), runs=50)
        
        # Export and benchmark ONNX FP32
        onnx_fp32_path = Path(output_dir) / "model_fp32.onnx"
        self.export_onnx_fp32(str(onnx_fp32_path))
        results['onnx_fp32'] = self.benchmark_model(str(onnx_fp32_path), runs=100)
        
        # Export and benchmark ONNX FP16
        onnx_fp16_path = Path(output_dir) / "model_fp16.onnx"
        self.export_onnx_fp16(str(onnx_fp16_path))
        results['onnx_fp16'] = self.benchmark_model(str(onnx_fp16_path), runs=100)
        
        # Try INT8 quantization
        try:
            onnx_int8_path = Path(output_dir) / "model_int8.onnx"
            self.quantize_int8(str(onnx_fp32_path), str(onnx_int8_path))
            results['onnx_int8'] = self.benchmark_model(str(onnx_int8_path), runs=100)
        except:
            print("⚠️  INT8 quantization skipped")
        
        # Save comparison results
        comparison_path = Path(output_dir) / "format_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Comparison saved to: {comparison_path}")
        
        # Print summary
        self.print_comparison_summary(results)
        
        return results
    
    def print_comparison_summary(self, results: Dict[str, Dict]):
        """Print comparison summary"""
        
        print(f"\n📈 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"{'Format':<12} {'Latency (ms)':<15} {'Speedup':<10} {'FPS':<10}")
        print("-" * 50)
        
        baseline = results.get('pytorch', {}).get('average_latency_ms', 1)
        
        for format_name, format_results in results.items():
            latency = format_results.get('average_latency_ms', 0)
            fps = format_results.get('throughput_fps', 0)
            speedup = baseline / latency if latency > 0 else 0
            
            print(f"{format_name:<12} {latency:<15.2f} {speedup:<10.2f}x {fps:<10.1f}")
        
        print("=" * 50)
    
    def export_all(self, output_dir: str, formats: List[str] = None):
        """Export to all specified formats"""
        
        if formats is None:
            formats = ['onnx_fp32', 'onnx_fp16', 'int8']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Exporting to {len(formats)} formats")
        print(f"📁 Output directory: {output_path}")
        
        exports = {}
        
        # Export to each format
        for fmt in formats:
            try:
                if fmt == 'onnx_fp32':
                    out_path = output_path / "model_fp32.onnx"
                    exports['onnx_fp32'] = self.export_onnx_fp32(str(out_path))
                
                elif fmt == 'onnx_fp16':
                    out_path = output_path / "model_fp16.onnx"
                    exports['onnx_fp16'] = self.export_onnx_fp16(str(out_path))
                
                elif fmt == 'int8':
                    # Need FP32 model first
                    fp32_path = output_path / "model_fp32.onnx"
                    if not fp32_path.exists():
                        self.export_onnx_fp32(str(fp32_path))
                    
                    out_path = output_path / "model_int8.onnx"
                    int8_path = self.quantize_int8(str(fp32_path), str(out_path))
                    if int8_path:
                        exports['int8'] = int8_path
                
                elif fmt == 'tensorrt':
                    print("⚠️  TensorRT export requires additional setup")
                    print("   Consider using trtexec or TensorRT Python API")
                
                else:
                    print(f"⚠️  Unknown format: {fmt}")
                    
            except Exception as e:
                print(f"❌ Failed to export {fmt}: {e}")
        
        # Save export manifest
        manifest = {
            'source_model': str(self.model_path),
            'exports': exports,
            'timestamp': datetime.now().isoformat(),
            'gpu_stats': self.gpu_monitor.get_performance_summary()
        }
        
        manifest_path = output_path / "export_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ All exports completed!")
        print(f"📄 Manifest saved: {manifest_path}")
        
        return exports

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Production Export for RTX 4060")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint (.pt)")
    
    parser.add_argument("--output-dir", type=str, default="models/production",
                       help="Output directory")
    
    parser.add_argument("--formats", type=str, nargs='+',
                       default=['onnx_fp32', 'onnx_fp16'],
                       help="Formats to export (onnx_fp32, onnx_fp16, int8)")
    
    parser.add_argument("--benchmark", action="store_true", default=True,
                       help="Run benchmarks after export")
    
    parser.add_argument("--compare", action="store_true", default=True,
                       help="Compare performance of different formats")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📦 ULTIMATE PRODUCTION EXPORT - RTX 4060")
    print("=" * 70)
    
    # Initialize exporter
    exporter = UltimateProductionExporter(args.model)
    
    # Export to specified formats
    exports = exporter.export_all(args.output_dir, args.formats)
    
    # Run benchmarks if requested
    if args.benchmark:
        print(f"\n🏎️  RUNNING BENCHMARKS")
        print("=" * 50)
        
        benchmark_results = {}
        for format_name, export_path in exports.items():
            if export_path:
                print(f"\n📊 Benchmarking {format_name}...")
                benchmark_results[format_name] = exporter.benchmark_model(
                    export_path, 
                    runs=100
                )
        
        # Save benchmark results
        benchmark_path = Path(args.output_dir) / "benchmark_results.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n📊 Benchmark results saved: {benchmark_path}")
    
    # Compare formats if requested
    if args.compare and len(exports) > 1:
        exporter.compare_formats(args.output_dir)
    
    print(f"\n🎉 Export pipeline complete!")
    print(f"📁 Check results in: {args.output_dir}")

if __name__ == "__main__":
    main()