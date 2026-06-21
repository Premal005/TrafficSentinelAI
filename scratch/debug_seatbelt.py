import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import math
from core.entity_detector import EntityDetector

detector = EntityDetector()
image_path = 'data/sample_images/wrong_side_sample.png'
image = cv2.imread(image_path)

# Run direct detection to get car and person nodes
dets = detector.detect(image)
print("Detections:")
for d in dets:
    print(f"  {d.class_name} @ {d.bbox} conf={d.confidence:.2f}")

# Find person crops that are inside the car
from core.scene_graph import is_four_wheeler_occupant
car_box = None
for d in dets:
    if d.class_name == "car":
        car_box = d.bbox
        break

if car_box:
    print(f"\nCar box: {car_box}")
    person_crops = []
    for d in dets:
        if d.class_name == "person" and is_four_wheeler_occupant(d.bbox, car_box):
            person_crops.append(d.bbox)
            print(f"  Occupant: {d.bbox}")
            
    # Run seatbelt heuristic step-by-step
    for idx, crop_box in enumerate(person_crops):
        x1, y1, x2, y2 = crop_box
        person_h = y2 - y1
        y1_torso = y1 + int(person_h * 0.30)
        y2_torso = y1 + int(person_h * 0.85)
        
        crop = image[y1_torso:y2_torso, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        min_line_len = max(15, int(crop.shape[0] * 0.3))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=min_line_len, maxLineGap=10)
        
        print(f"\nOccupant {idx}:")
        print(f"  Torso shape: {crop.shape}")
        print(f"  Min line length: {min_line_len}")
        if lines is not None:
            print(f"  Found {len(lines)} lines:")
            angles = []
            matching_lines = []
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
                angles.append(angle)
                if 25 < abs(angle) < 65:
                    matching_lines.append(line[0])
            print(f"    Angles: {[round(a, 1) for a in angles[:10]]}...")
            print(f"    Matching lines count: {len(matching_lines)}")
        else:
            print("  No lines found by HoughLinesP")
else:
    print("No car found in image.")
