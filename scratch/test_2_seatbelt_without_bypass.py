import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import math
from core.entity_detector import EntityDetector, Detection
from config.settings import EntityClass

detector = EntityDetector()
# Temporarily disable the cartoon check by modifying the detector's internal state or setting it to False
detector._is_cartoon_seatbelt_img = False

image_path = 'data/sample_images/2. seatbelt.png'
image = cv2.imread(image_path)

# Occupants of the Creta in 2. seatbelt.png:
driver_bbox = (344, 364, 500, 487)
passenger_bbox = (558, 379, 703, 489)

# Run the heuristic on these bboxes
results = detector.detect_seatbelts(image, [driver_bbox, passenger_bbox])

print("\n--- Seatbelt Detection results for 2. seatbelt.png without bypass ---")
for d in results:
    print(f"BBox: {d.bbox} -> Class: {d.class_name} (conf={d.confidence:.2f})")
