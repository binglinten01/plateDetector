#!/usr/bin/env python3
"""
Synthetic Data Generation Script - Simplified Version
Generates synthetic license plates for rare patterns and data balancing
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont
import yaml
import argparse

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def setup_logger(name: str):
    """Setup a simple logger."""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger("synthesizer")

class SimpleSynthesizer:
    """Simple synthetic data generator for license plates."""
    
    def __init__(self, config_path: str = "configs/data_license_plate.yaml"):
        """Initialize synthesizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.synthetic_dir = Path("datasets/06_synthetic")
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Vietnamese license plate patterns
        self.digits = '0123456789'
        self.vn_letters = 'ABCDEFGHKLMNPSTUVXYZ'
        
        logger.info(f"Initialized SimpleSynthesizer")
    
    def generate_plate_text(self):
        """Generate realistic Vietnamese license plate text."""
        patterns = [
            ("##A#####", 0.6),   # 51A-12345
            ("##A####", 0.2),    # 51A-1234
            ("##AB###", 0.1),    # 51AB-123
            ("###A####", 0.05),  # Special
            ("##A##", 0.03),     # Old format
            ("###A##", 0.02),    # Government
        ]
        
        patterns_text = [p[0] for p in patterns]
        weights = [p[1] for p in patterns]
        pattern = random.choices(patterns_text, weights=weights, k=1)[0]
        
        plate_text = ""
        for char in pattern:
            if char == '#':
                plate_text += random.choice(self.digits)
            elif char == 'A' or char == 'B':
                plate_text += random.choice(self.vn_letters)
            else:
                plate_text += char
        
        # Format with dash
        if len(plate_text) >= 5:
            dash_pos = 3 if plate_text[2] in self.vn_letters else 2
            formatted = plate_text[:dash_pos] + '-' + plate_text[dash_pos:]
        else:
            formatted = plate_text
        
        return formatted, pattern
    
    def create_plate_image(self, plate_text: str):
        """Create a simple license plate image."""
        # Choose plate style
        styles = [
            ('white', (255, 255, 255), (0, 0, 0)),
            ('yellow', (0, 255, 255), (0, 0, 0)),
            ('blue', (255, 0, 0), (255, 255, 255)),
            ('red', (0, 0, 255), (255, 255, 255)),
        ]
        
        style_name, bg_color, text_color = random.choice(styles)
        
        # Create plate
        width, height = 400, 150
        image = np.ones((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = bg_color[0]
        image[:, :, 1] = bg_color[1]
        image[:, :, 2] = bg_color[2]
        
        # Add border
        cv2.rectangle(image, (5, 5), (width-5, height-5), (0, 0, 0), 2)
        
        # Add text using PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Try to load a font
            font_size = 60
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), plate_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Draw text
            rgb_color = (text_color[2], text_color[1], text_color[0])
            draw.text((x, y), plate_text, font=font, fill=rgb_color)
            
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.warning(f"Could not add text with PIL: {e}")
            # Fallback: use OpenCV text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
            x = (width - text_size[0]) // 2
            y = (height + text_size[1]) // 2
            
            cv2.putText(image, plate_text, (x, y), font, font_scale, text_color, thickness)
        
        # Add some realism
        if random.random() < 0.3:
            # Add noise
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # Add blur
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Place on background
        bg_width = width * 2
        bg_height = height * 2
        background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * random.randint(50, 150)
        
        x_offset = random.randint(0, bg_width - width)
        y_offset = random.randint(0, bg_height - height)
        background[y_offset:y_offset+height, x_offset:x_offset+width] = image
        
        # Calculate bbox in YOLO format
        x_center = (x_offset + width/2) / bg_width
        y_center = (y_offset + height/2) / bg_height
        bbox_width = width / bg_width
        bbox_height = height / bg_height
        
        bbox = [x_center, y_center, bbox_width, bbox_height]
        
        return {
            'image': background,
            'text': plate_text,
            'bbox': bbox,
            'style': style_name
        }
    
    def generate_synthetic_dataset(self, n_plates: int = 1000):
        """Generate synthetic license plate dataset."""
        logger.info("=" * 60)
        logger.info("STARTING SYNTHETIC DATA GENERATION")
        logger.info("=" * 60)
        
        # Create output directories
        images_dir = self.synthetic_dir / "images"
        labels_dir = self.synthetic_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        generation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_generated': 0,
            'failed_generations': 0,
            'generated_plates': []
        }
        
        # Generate plates
        for plate_idx in range(n_plates):
            try:
                # Generate plate
                plate_text, pattern = self.generate_plate_text()
                plate_data = self.create_plate_image(plate_text)
                
                # Save image
                plate_filename = f"synthetic_plate_{plate_idx:06d}.jpg"
                plate_path = images_dir / plate_filename
                cv2.imwrite(str(plate_path), plate_data['image'])
                
                # Save label
                label_path = plate_path.with_suffix('.txt')
                with open(label_path, 'w') as f:
                    bbox = plate_data['bbox']
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                # Update report
                generation_report['total_generated'] += 1
                
                if plate_idx < 50:  # Store first 50 for report
                    generation_report['generated_plates'].append({
                        'filename': plate_filename,
                        'text': plate_text,
                        'pattern': pattern
                    })
                
                if (plate_idx + 1) % 100 == 0:
                    logger.info(f"Generated {plate_idx + 1} plates...")
                    
            except Exception as e:
                logger.error(f"Error generating plate {plate_idx}: {e}")
                generation_report['failed_generations'] += 1
        
        # Create data.yaml
        data_yaml = {
            'path': str(self.synthetic_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': 1,
            'names': ['license_plate']
        }
        
        yaml_path = self.synthetic_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Save report
        report_path = self.synthetic_dir / "synthetic_generation_report.json"
        with open(report_path, 'w') as f:
            json.dump(generation_report, f, indent=2)
        
        # Print summary
        self._print_summary(generation_report)
        
        return generation_report
    
    def _print_summary(self, report: dict):
        """Print generation summary."""
        print("\n" + "=" * 70)
        print("SYNTHETIC DATA GENERATION - SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 GENERATION STATISTICS")
        print(f"   Total plates generated: {report['total_generated']:,}")
        print(f"   Failed generations: {report.get('failed_generations', 0)}")
        
        print(f"\n📁 OUTPUT LOCATION")
        print(f"   Dataset: {self.synthetic_dir}")
        print(f"   Images: {self.synthetic_dir / 'images'}")
        print(f"   Labels: {self.synthetic_dir / 'labels'}")
        print(f"   Report: {self.synthetic_dir / 'synthetic_generation_report.json'}")
        
        print(f"\n💡 USAGE SUGGESTIONS")
        print("   1. Mix with real dataset for training")
        print("   2. Use for testing rare patterns")
        print("   3. Combine with augmented dataset")
        
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic license plate data"
    )
    parser.add_argument(
        "--config", 
        default="configs/data_license_plate.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--n-plates", 
        type=int,
        default=1000,
        help="Number of synthetic plates to generate"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate previews only"
    )
    parser.add_argument(
        "--n-previews",
        type=int,
        default=5,
        help="Number of previews to generate"
    )
    
    args = parser.parse_args()
    
    synthesizer = SimpleSynthesizer(args.config)
    
    if args.preview:
        # Generate previews
        preview_dir = Path("results/synthetic_previews")
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {args.n_previews} previews...")
        for i in range(args.n_previews):
            try:
                plate_text, pattern = synthesizer.generate_plate_text()
                plate_data = synthesizer.create_plate_image(plate_text)
                
                preview_path = preview_dir / f"preview_{i:02d}_{plate_text.replace('-', '_')}.jpg"
                cv2.imwrite(str(preview_path), plate_data['image'])
                
                print(f"✅ Preview {i+1}: {plate_text}")
                
            except Exception as e:
                print(f"❌ Failed to generate preview {i+1}: {e}")
        
        print(f"\n🎨 Generated {args.n_previews} previews in: {preview_dir}")
    
    else:
        # Generate full dataset
        report = synthesizer.generate_synthetic_dataset(args.n_plates)
        print(f"\n✅ Generated {report['total_generated']} synthetic plates")
        print(f"📁 Output: {synthesizer.synthetic_dir}")

if __name__ == "__main__":
    main()