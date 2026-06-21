import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    for p in PROJECT_ROOT.glob("crop_*.png"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        center_hsv = hsv[h//2, w//2]
        center_bgr = img[h//2, w//2]
        print(f"Crop {p.name}:")
        print(f"  Center BGR: {center_bgr}")
        print(f"  Center HSV: {center_hsv}")

if __name__ == "__main__":
    main()
