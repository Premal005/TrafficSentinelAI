import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from core.entity_detector import EntityDetector
from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
from config.settings import EntityClass, TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES

detector = EntityDetector()
image_path = 'data/sample_images/wrong_side_sample.png'
image = cv2.imread(image_path)

detections = detector.detect(image)
rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]

seatbelt_crops = []
for r in rider_nodes:
    is_occupant = False
    for v in vehicles:
        if v.entity_class in FOUR_WHEELER_CLASSES:
            if is_four_wheeler_occupant(r.bbox, v.bbox):
                is_occupant = True
                print(f"Person {r.bbox} is occupant of vehicle {v.bbox}")
    if is_occupant:
        seatbelt_crops.append(r.bbox)

if seatbelt_crops:
    seatbelt_detections = detector.detect_seatbelts(image, seatbelt_crops)
    print("\nSeatbelt detections generated:")
    for d in seatbelt_detections:
        print(f"  {d.class_name} @ {d.bbox} conf={d.confidence:.2f}")
else:
    print("No seatbelt crops found.")
