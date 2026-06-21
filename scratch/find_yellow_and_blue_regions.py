import cv2
import numpy as np
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    # Yellow (for yellow shirt)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    # Blue (for blue shirt)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    # Purple/Violet (for purple shirt)
    lower_purple = np.array([135, 50, 50])
    upper_purple = np.array([165, 255, 255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Find bounding box of connected components for each mask
    for color_name, mask in [("Yellow", mask_yellow), ("Blue", mask_blue), ("Purple", mask_purple)]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        print(f"\n--- {color_name} Regions ---")
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:  # ignore tiny noise
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                print(f"Region {i}: BBox=({x}, {y}, {x+w}, {y+h}), Area={area}")

if __name__ == "__main__":
    main()
