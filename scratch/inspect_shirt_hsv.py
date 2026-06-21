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
        h, w = img.shape[:2]
        # Inspect lower part of the crop (e.g., y=0.75*h)
        torso_y = int(h * 0.75)
        torso_x = w // 2
        pixel_bgr = img[torso_y, torso_x]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        pixel_hsv = hsv[torso_y, torso_x]
        print(f"Crop {p.name}:")
        print(f"  Torso BGR: {pixel_bgr}")
        print(f"  Torso HSV: {pixel_hsv}")

if __name__ == "__main__":
    main()
