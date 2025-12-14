#!/usr/bin/env python3
"""
Ultimate Dataset Downloader for Vietnam License Plate Recognition
Uses Roboflow SDK and Kaggle CLI
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import json
from dotenv import load_dotenv

# ================= LOAD ENVIRONMENT VARIABLES =================
load_dotenv()

# Get the absolute path to the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logging import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
# ==============================================================

# Setup
RAW_DIR = Path("datasets/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger("DatasetDownloader")

# Dataset configurations - UPDATED FOR ROBOTFLOW SDK
DATASETS = {
    "roboflow_lpr": {
        "type": "roboflow_sdk",
        "workspace": "roboflow-universe-projects",
        "project": "license-plate-recognition-rxg4e",
        "version": 4,
        "format": "yolov11",  # <-- ĐÚNG format từ code web
        "api_key_env": "ROBOFLOW_API_KEY",
        "expected_size": "3.5GB",
        "description": "License Plate Recognition dataset"
    },
    "kaggle_vn": {
        "type": "kaggle",
        "slug": "bomaich/vnlicenseplate",
        "format": "yolo",
        "expected_size": "500MB",
        "description": "Vietnam License Plates from Kaggle"
    },
    "roboflow_vietnam": {
        "type": "roboflow_sdk", 
        "workspace": "tran-ngoc-xuan-tin-k15-hcm-dpuid",
        "project": "vietnam-license-plate-h8t3n",
        "version": 1,
        "format": "coco",
        "api_key_env": "ROBOFLOW_API_KEY",
        "expected_size": "1.2GB",
        "description": "Vietnam License Plate dataset"
    }
}

class DatasetDownloader:
    def __init__(self):
        self.check_credentials()
        self.ensure_roboflow_sdk()
        
    def check_credentials(self):
        """Check if required API keys are available"""
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            logger.info(f"✓ Roboflow API Key loaded: {masked_key}")
        else:
            logger.error("❌ ROBOFLOW_API_KEY not found in .env file")
            logger.info("Add to .env: ROBOFLOW_API_KEY=wEQ05QWIHhMkk39PG4mb")
            return False
        
        # Check Kaggle
        kaggle_dir = Path.home() / ".kaggle"
        if (kaggle_dir / "kaggle.json").exists():
            logger.info("✓ Kaggle credentials found")
        else:
            logger.warning("⚠ Kaggle API credentials not found")
            logger.info("Kaggle datasets will use CLI (if installed)")
        
        return True
    
    def ensure_roboflow_sdk(self):
        """Ensure Roboflow SDK is installed"""
        try:
            import roboflow
            logger.info("✓ Roboflow SDK already installed")
        except ImportError:
            logger.warning("Roboflow SDK not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
                logger.info("✓ Roboflow SDK installed successfully")
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to install Roboflow SDK")
                logger.info("Install manually: pip install roboflow")
                return False
        return True
    
    def download_roboflow_sdk(self, dataset_name, config):
        """Download dataset using Roboflow Python SDK (đúng như web hướng dẫn)"""
        logger.info(f"Downloading {dataset_name} using Roboflow SDK...")
        
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.error("❌ ROBOFLOW_API_KEY not set")
            return False
        
        try:
            from roboflow import Roboflow
            
            # Get config values
            workspace = config["workspace"]
            project_name = config["project"]
            version_num = config["version"]
            format_type = config["format"]
            
            logger.info(f"  Workspace: {workspace}")
            logger.info(f"  Project: {project_name}")
            logger.info(f"  Version: {version_num}")
            logger.info(f"  Format: {format_type}")
            
            # Initialize Roboflow (đúng như code web)
            rf = Roboflow(api_key=api_key)
            
            # Get project và version (đúng như code web)
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            
            # Tạo output directory
            output_dir = RAW_DIR / dataset_name
            output_dir.mkdir(exist_ok=True)
            
            logger.info(f"  Downloading to: {output_dir}")
            
            # Download dataset (đúng như code web - yolov11)
            # Note: Nếu yolov11 không hoạt động, thử yolov8
            try:
                dataset = version.download(format_type)
                logger.info(f"✓ Downloaded with format: {format_type}")
            except Exception as format_error:
                logger.warning(f"Format {format_type} failed, trying yolov8...")
                try:
                    dataset = version.download("yolov8")
                    logger.info("✓ Downloaded with format: yolov8")
                except Exception as fallback_error:
                    logger.error(f"All format attempts failed: {fallback_error}")
                    return False
            
            # Kiểm tra downloaded location
            downloaded_path = Path(dataset.location)
            if not downloaded_path.exists():
                logger.error(f"❌ Downloaded path not found: {downloaded_path}")
                return False
            
            logger.info(f"✓ Dataset downloaded to: {downloaded_path}")
            
            # Di chuyển files đến output directory của chúng ta
            self.move_downloaded_files(downloaded_path, output_dir)
            
            # Xóa thư mục tạm nếu còn
            if downloaded_path.exists():
                try:
                    shutil.rmtree(downloaded_path)
                except:
                    pass
            
            logger.info(f"✅ Successfully downloaded {dataset_name}")
            return True
            
        except ImportError:
            logger.error("❌ Roboflow SDK not installed")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    def move_downloaded_files(self, source_dir, target_dir):
        """Move files from downloaded directory to our organized directory"""
        try:
            # Đếm và di chuyển files
            moved_files = 0
            
            for item in source_dir.rglob("*"):
                if item.is_file():
                    # Tạo relative path
                    rel_path = item.relative_to(source_dir)
                    target_path = target_dir / rel_path
                    
                    # Tạo thư mục đích nếu chưa có
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Di chuyển file
                    shutil.move(str(item), str(target_path))
                    moved_files += 1
            
            logger.info(f"✓ Moved {moved_files} files to {target_dir}")
            
        except Exception as e:
            logger.error(f"Error moving files: {e}")
    
    def download_kaggle(self, dataset_name, config):
        """Download dataset from Kaggle"""
        logger.info(f"Downloading {dataset_name} from Kaggle...")
        
        try:
            output_dir = RAW_DIR / dataset_name
            output_dir.mkdir(exist_ok=True)
            
            cmd = [
                "kaggle", "datasets", "download",
                "-d", config["slug"],
                "-p", str(output_dir),
                "--unzip"
            ]
            
            logger.info(f"  Running: kaggle datasets download -d {config['slug']}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minutes
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded {dataset_name}")
                return True
            else:
                logger.error(f"❌ Kaggle download failed: {result.stderr[:200]}")
                # Thử cách khác
                return self.download_kaggle_alternative(dataset_name, config)
                
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    def download_kaggle_alternative(self, dataset_name, config):
        """Alternative Kaggle download method"""
        try:
            logger.info(f"Trying alternative download for {dataset_name}")
            
            output_dir = RAW_DIR / dataset_name
            
            # Clean any existing files
            for f in output_dir.glob("*"):
                if f.is_file():
                    f.unlink()
            
            # Try without --unzip first
            cmd = [
                "kaggle", "datasets", "download",
                "-d", config["slug"],
                "-p", str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Tìm và extract zip
                zip_files = list(output_dir.glob("*.zip"))
                if zip_files:
                    import zipfile
                    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    zip_files[0].unlink()
                    logger.info(f"✓ Downloaded and extracted {dataset_name}")
                    return True
            
            return False
                
        except Exception as e:
            logger.error(f"Alternative download failed: {e}")
            return False
    
    def verify_download(self, dataset_name):
        """Verify downloaded dataset integrity"""
        dataset_dir = RAW_DIR / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        # Count files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']
        label_extensions = ['.txt', '.json', '.xml', '.csv']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.rglob(f"*{ext}"))
            image_files.extend(dataset_dir.rglob(f"*{ext.upper()}"))
        
        label_files = []
        for ext in label_extensions:
            label_files.extend(dataset_dir.rglob(f"*{ext}"))
            label_files.extend(dataset_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates
        image_files = list(set(image_files))
        label_files = list(set(label_files))
        
        if len(image_files) == 0:
            logger.warning(f"⚠ No images found in {dataset_name}")
            return False
        
        logger.info(f"✓ {dataset_name}: {len(image_files)} images, {len(label_files)} labels")
        
        # Hiển thị vài file mẫu
        if image_files:
            sample_count = min(3, len(image_files))
            logger.info(f"  Sample files: {[f.name for f in image_files[:sample_count]]}")
        
        return True
    
    def count_total_images(self):
        """Count total images in all datasets"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']
        total = 0
        
        for dataset_dir in RAW_DIR.iterdir():
            if dataset_dir.is_dir():
                for ext in image_extensions:
                    total += len(list(dataset_dir.rglob(f"*{ext}")))
                    total += len(list(dataset_dir.rglob(f"*{ext.upper()}")))
        
        return total
    
    def create_summary(self):
        """Create summary JSON file"""
        summary = {}
        total_images = 0
        total_size_mb = 0
        
        for dataset_name, config in DATASETS.items():
            dataset_dir = RAW_DIR / dataset_name
            if dataset_dir.exists() and dataset_dir.is_dir():
                # Count images
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(dataset_dir.rglob(f"*{ext}"))
                image_files = list(set(image_files))
                
                # Calculate size
                size_bytes = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
                
                summary[dataset_name] = {
                    "images": len(image_files),
                    "size_mb": round(size_bytes / (1024*1024), 1),
                    "format": config.get("format", "unknown"),
                    "description": config.get("description", "")
                }
                
                total_images += len(image_files)
                total_size_mb += size_bytes / (1024*1024)
        
        summary["TOTAL"] = {
            "images": total_images,
            "size_mb": round(total_size_mb, 1),
            "datasets": len([k for k in summary.keys() if k != "TOTAL"])
        }
        
        # Save summary
        summary_file = RAW_DIR / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Summary saved to: {summary_file}")
        return summary
    
    def run(self):
        """Main download pipeline"""
        logger.info("=" * 60)
        logger.info("DATASET DOWNLOAD PIPELINE")
        logger.info("=" * 60)
        
        # Kiểm tra credentials
        if not self.check_credentials():
            logger.error("❌ Missing required credentials")
            return
        
        # Clean raw directory
        if RAW_DIR.exists():
            logger.info("🧹 Cleaning raw directory...")
            shutil.rmtree(RAW_DIR)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        successful = []
        failed = []
        
        # Download từng dataset
        for dataset_name, config in DATASETS.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"📦 Processing: {dataset_name}")
            logger.info(f"{'='*40}")
            logger.info(f"Description: {config.get('description', '')}")
            
            success = False
            
            if config["type"] == "roboflow_sdk":
                success = self.download_roboflow_sdk(dataset_name, config)
            elif config["type"] == "kaggle":
                success = self.download_kaggle(dataset_name, config)
            
            # Verify
            if success:
                if self.verify_download(dataset_name):
                    successful.append(dataset_name)
                    logger.info(f"✅ SUCCESS: {dataset_name}")
                else:
                    failed.append(dataset_name)
                    logger.warning(f"⚠ Downloaded but verification failed: {dataset_name}")
            else:
                failed.append(dataset_name)
                logger.error(f"❌ FAILED: {dataset_name}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        
        if successful:
            summary = self.create_summary()
            total_images = summary["TOTAL"]["images"]
            
            logger.info(f"✅ Successful: {len(successful)}/{len(DATASETS)} datasets")
            logger.info(f"❌ Failed: {len(failed)} datasets")
            
            logger.info("\n📊 DATASET DETAILS:")
            logger.info("-" * 40)
            
            for name, stats in summary.items():
                if name != "TOTAL":
                    logger.info(f"{name}:")
                    logger.info(f"  Images: {stats['images']}")
                    logger.info(f"  Size: {stats['size_mb']} MB")
                    logger.info(f"  Format: {stats['format']}")
            
            logger.info("\n" + "=" * 40)
            logger.info(f"🎯 TOTAL: {total_images} images")
            logger.info(f"💾 Total size: {summary['TOTAL']['size_mb']} MB")
            logger.info(f"📁 Location: {RAW_DIR.absolute()}")
            logger.info("=" * 40)
            
            # Recommendation
            if total_images < 1000:
                logger.warning(f"\n⚠ WARNING: Only {total_images} images")
                logger.info("Recommended next steps:")
                logger.info("1. Run Kaggle downloads manually")
                logger.info("2. Add more Kaggle datasets to config")
                logger.info("3. Use data augmentation (script 05)")
            elif total_images < 3000:
                logger.info(f"\n✓ Good start: {total_images} images")
                logger.info("You can begin training with augmentation")
            else:
                logger.info(f"\n🎉 Excellent: {total_images} images")
                logger.info("Ready for model training!")
                
        else:
            logger.error("❌ No datasets were successfully downloaded")
            logger.info("\n💡 Troubleshooting:")
            logger.info("1. Check internet connection")
            logger.info("2. Verify ROBOFLOW_API_KEY in .env file")
            logger.info("3. Install Kaggle CLI: pip install kaggle")

def main():
    downloader = DatasetDownloader()
    downloader.run()

if __name__ == "__main__":
    main()