import sys
from pathlib import Path
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS
from core.entity_detector import EntityDetector

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "1. helmet.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    image = cv2.imread(str(image_path))
    detector = EntityDetector(SETTINGS)
    
    results = detector._vehicle_model(
        image,
        conf=0.15,
        verbose=False,
    )
    
    print("\n--- Detections from 1. helmet.png (conf >= 0.15) ---")
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy()
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                print(f"Class={cls_name} ({cls_id}), Conf={conf:.3f}, BBox={tuple(int(c) for c in xyxy)}")

if __name__ == "__main__":
    main()
