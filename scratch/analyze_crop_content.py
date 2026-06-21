import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_crop(crop_path):
    img = cv2.imread(str(crop_path))
    if img is None:
        return
    avg_color_bgr = np.mean(img, axis=(0, 1))
    print(f"File: {crop_path.name}, Size: {img.shape}")
    print(f"  Avg BGR: {avg_color_bgr}")

def main():
    for p in PROJECT_ROOT.glob("crop_*.png"):
        analyze_crop(p)

if __name__ == "__main__":
    main()
