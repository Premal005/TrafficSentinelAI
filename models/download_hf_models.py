"""
Download pre-trained models from HuggingFace for TrafficSentinel AI.

Downloads:
  - Helmet detection: keremberke/yolov8s-hard-hat-detection
  - License plate detection: Koushim/yolov8-license-plate-detection

Usage:
    python models/download_hf_models.py
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"


def download_helmet_model():
    """Download the helmet detection model from HuggingFace."""
    target = WEIGHTS_DIR / "yolov8s_helmet.pt"
    if target.exists():
        logger.info("✅ Helmet model already exists at %s", target)
        return True

    logger.info("Downloading helmet detection model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="keremberke/yolov8s-hard-hat-detection",
            filename="best.pt",
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        # Rename to our expected filename
        dl_path = Path(downloaded)
        if dl_path.exists():
            dl_path.rename(target)
            logger.info("✅ Helmet model saved to %s", target)
            return True
        else:
            logger.error("Download succeeded but file not found at %s", downloaded)
            return False
    except Exception as e:
        logger.error("❌ Failed to download helmet model: %s", e)
        # Fallback: try direct ultralytics hub load
        try:
            logger.info("Trying ultralytics hub fallback...")
            from ultralytics import YOLO
            model = YOLO("keremberke/yolov8s-hard-hat-detection")
            # Export/save the model
            import shutil
            # The model file should be cached by ultralytics
            model_path = Path(model.ckpt_path) if hasattr(model, 'ckpt_path') else None
            if model_path and model_path.exists():
                shutil.copy2(str(model_path), str(target))
                logger.info("✅ Helmet model saved via ultralytics to %s", target)
                return True
        except Exception as e2:
            logger.error("❌ Ultralytics fallback also failed: %s", e2)
        return False


def download_plate_model():
    """Download the license plate detection model from HuggingFace."""
    target = WEIGHTS_DIR / "yolov8s_plate.pt"
    if target.exists():
        logger.info("✅ Plate model already exists at %s", target)
        return True

    logger.info("Downloading license plate detection model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Koushim/yolov8-license-plate-detection",
            filename="best.pt",
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        dl_path = Path(downloaded)
        if dl_path.exists():
            dl_path.rename(target)
            logger.info("✅ Plate model saved to %s", target)
            return True
        else:
            logger.error("Download succeeded but file not found at %s", downloaded)
            return False
    except Exception as e:
        logger.error("❌ Failed to download plate model: %s", e)
        try:
            logger.info("Trying ultralytics hub fallback...")
            from ultralytics import YOLO
            model = YOLO("Koushim/yolov8-license-plate-detection")
            import shutil
            model_path = Path(model.ckpt_path) if hasattr(model, 'ckpt_path') else None
            if model_path and model_path.exists():
                shutil.copy2(str(model_path), str(target))
                logger.info("✅ Plate model saved via ultralytics to %s", target)
                return True
        except Exception as e2:
            logger.error("❌ Ultralytics fallback also failed: %s", e2)
        return False


def verify_models():
    """Verify downloaded models load correctly and print class names."""
    from ultralytics import YOLO

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFYING DOWNLOADED MODELS")
    logger.info("=" * 60)

    helmet_path = WEIGHTS_DIR / "yolov8s_helmet.pt"
    plate_path = WEIGHTS_DIR / "yolov8s_plate.pt"

    if helmet_path.exists():
        try:
            model = YOLO(str(helmet_path))
            logger.info("✅ Helmet model classes: %s", model.names)
        except Exception as e:
            logger.error("❌ Helmet model failed to load: %s", e)
    else:
        logger.warning("⚠️ Helmet model not found at %s", helmet_path)

    if plate_path.exists():
        try:
            model = YOLO(str(plate_path))
            logger.info("✅ Plate model classes: %s", model.names)
        except Exception as e:
            logger.error("❌ Plate model failed to load: %s", e)
    else:
        logger.warning("⚠️ Plate model not found at %s", plate_path)


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("🚦 TrafficSentinel AI — HuggingFace Model Downloader")
    logger.info("   Target directory: %s", WEIGHTS_DIR)
    logger.info("")

    h_ok = download_helmet_model()
    p_ok = download_plate_model()

    if h_ok and p_ok:
        verify_models()
        logger.info("")
        logger.info("🎉 All models downloaded and verified!")
    else:
        logger.warning("⚠️ Some models failed to download. Check errors above.")


if __name__ == "__main__":
    main()
