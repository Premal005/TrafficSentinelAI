import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_color(crop_path):
    img = cv2.imread(str(crop_path))
    if img is None:
        return "Unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Yellow
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    # Blue
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])
    # Purple/Violet
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    
    yellow_pixels = np.sum(mask_yellow > 0)
    blue_pixels = np.sum(mask_blue > 0)
    purple_pixels = np.sum(mask_purple > 0)
    
    total = img.shape[0] * img.shape[1]
    print(f"File: {crop_path.name}")
    print(f"  Yellow pixels: {yellow_pixels} ({yellow_pixels/total:.1%})")
    print(f"  Blue pixels: {blue_pixels} ({blue_pixels/total:.1%})")
    print(f"  Purple pixels: {purple_pixels} ({purple_pixels/total:.1%})")

def main():
    for p in PROJECT_ROOT.glob("crop_*.png"):
        analyze_color(p)

if __name__ == "__main__":
    main()
