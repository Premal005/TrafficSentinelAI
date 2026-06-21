import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    img = cv2.imread(str(image_path))
    
    crops = {
        "crop_164_350_249_449": (164, 350, 249, 449),
        "crop_344_364_500_487": (344, 364, 500, 487),
        "crop_558_379_703_489": (558, 379, 703, 489)
    }
    
    for name, bbox in crops.items():
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        output_path = PROJECT_ROOT / f"{name}.png"
        cv2.imwrite(str(output_path), crop)
        print(f"Saved {name}.png")

if __name__ == "__main__":
    main()
