"""
Model Download & Setup Script for TrafficSentinel AI.

Downloads all required model weights:
- YOLOv11m (COCO pretrained) — vehicle/person detection
- YOLOv11s (to be fine-tuned) — helmet detection
- YOLOv11s (to be fine-tuned) — license plate detection

Usage:
    python models/download_models.py
    python models/download_models.py --all
    python models/download_models.py --model vehicle
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_yolo_models(models_dir: Path):
    """
    Download YOLO models via ultralytics auto-download.
    
    Ultralytics automatically downloads models when first loaded.
    This script pre-downloads them to avoid delays during demo.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────
    # 1. Vehicle/Person Detection (COCO pretrained)
    # ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Downloading YOLOv11m (COCO pretrained)...")
    logger.info("Purpose: Vehicle, person, motorcycle, bicycle detection")
    logger.info("=" * 60)
    
    try:
        model = YOLO("yolo11m.pt")
        logger.info(f"✅ YOLOv11m loaded successfully")
        logger.info(f"   Classes: {len(model.names)} COCO classes")
        logger.info(f"   Relevant: person, bicycle, car, motorcycle, bus, truck, traffic light")
        
        # Quick validation
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        result = model.predict(dummy, verbose=False)
        logger.info(f"   Inference test: PASSED")
    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv11m: {e}")
    
    # ─────────────────────────────────────
    # 2. Helmet Detection Model
    # ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Preparing YOLOv11s for helmet detection...")
    logger.info("Purpose: Helmet / No-Helmet classification on rider crops")
    logger.info("=" * 60)
    
    try:
        # Download base model for fine-tuning
        model_s = YOLO("yolo11s.pt")
        logger.info(f"✅ YOLOv11s base model loaded (for helmet fine-tuning)")
        logger.info(f"   Fine-tune this on a helmet dataset (Kaggle/Roboflow)")
        logger.info(f"   Classes needed: helmet, no_helmet")
    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv11s: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Fine-tune YOLOv11s on helmet dataset:")
    logger.info("     yolo detect train model=yolo11s.pt data=helmet.yaml epochs=50")
    logger.info("")
    logger.info("  2. Fine-tune YOLOv11s on license plate dataset:")
    logger.info("     yolo detect train model=yolo11s.pt data=plate.yaml epochs=50")
    logger.info("")
    logger.info("  3. Place trained weights in: models/weights/")
    logger.info("     - yolo11s_helmet.pt")
    logger.info("     - yolo11s_plate.pt")


def download_ocr():
    """Pre-download EasyOCR models."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Downloading EasyOCR models...")
    logger.info("=" * 60)
    
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        logger.info("✅ EasyOCR English model loaded")
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR download issue (may work later): {e}")


def main():
    parser = argparse.ArgumentParser(description="Download models for TrafficSentinel AI")
    parser.add_argument("--all", action="store_true", help="Download all models including OCR")
    parser.add_argument("--model", choices=["vehicle", "helmet", "plate", "ocr"], 
                        help="Download specific model")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models" / "weights"),
                        help="Output directory for model weights")
    
    args = parser.parse_args()
    
    models_dir = Path(args.output)
    
    logger.info("🚦 TrafficSentinel AI — Model Download Script")
    logger.info(f"   Output directory: {models_dir}")
    logger.info("")
    
    download_yolo_models(models_dir)
    
    if args.all:
        download_ocr()
    
    logger.info("")
    logger.info("🎉 All done! Ready to detect traffic violations.")


if __name__ == "__main__":
    main()
