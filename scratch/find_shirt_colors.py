import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def scan_y(img, y):
    print(f"\n--- Scanning y={y} ---")
    for x in range(167, 865, 20):
        bgr = img[y, x]
        print(f"x={x}: {list(bgr)}")

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    img = cv2.imread(str(image_path))
    scan_y(img, 500)
    scan_y(img, 550)

if __name__ == "__main__":
    main()
