import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import math

image_path = 'data/sample_images/wrong_side_sample.png'
image = cv2.imread(image_path)

# Driver and Passenger bboxes from detector output
driver_bbox = (404, 483, 520, 584)
passenger_bbox = (547, 491, 660, 582)

def test_heuristic(bbox, label, x_crop_ratio=1.0):
    x1, y1, x2, y2 = bbox
    person_h = y2 - y1
    person_w = x2 - x1
    
    # Crop horizontally if ratio < 1.0
    if x_crop_ratio < 1.0:
        center_x = (x1 + x2) // 2
        new_w = int(person_w * x_crop_ratio)
        x1_crop = max(x1, center_x - new_w // 2)
        x2_crop = min(x2, center_x + new_w // 2)
    else:
        x1_crop, x2_crop = x1, x2
        
    y1_torso = y1 + int(person_h * 0.30)
    y2_torso = y1 + int(person_h * 0.85)
    
    crop = image[y1_torso:y2_torso, x1_crop:x2_crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0) # try 3x3 instead of 5x5
    edges = cv2.Canny(blurred, 30, 100) # lower Canny thresholds
    
    min_line_len = max(10, int(crop.shape[0] * 0.25)) # lower min line length
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=min_line_len, maxLineGap=15)
    
    print(f"\n{label} (X-crop: {x_crop_ratio}):")
    print(f"  Crop shape: {crop.shape}")
    print(f"  Min line length: {min_line_len}")
    
    has_seatbelt = False
    if lines is not None:
        print(f"  Found {len(lines)} lines:")
        matching_lines = []
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
            # Also accept negative angles
            if 20 < abs(angle) < 70:
                has_seatbelt = True
                matching_lines.append((line[0], angle))
        print(f"    Matching lines: {len(matching_lines)}")
        for l, a in matching_lines[:5]:
            print(f"      Line: {l}, Angle: {a:.1f}")
    else:
        print("  No lines found")
    return has_seatbelt

print("--- Testing Heuristics ---")
for r in [1.0, 0.8, 0.6, 0.5]:
    test_heuristic(driver_bbox, "Driver", r)
    test_heuristic(passenger_bbox, "Passenger", r)
