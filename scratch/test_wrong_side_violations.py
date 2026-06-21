import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from core.entity_detector import EntityDetector
from core.scene_graph import SceneGraph
from core.violation_engine import ViolationEngine
from config.settings import SETTINGS

detector = EntityDetector()
scene_graph_builder = SceneGraph(SETTINGS)
violation_engine = ViolationEngine(SETTINGS)

image = cv2.imread('data/sample_images/wrong_side_sample.png')
detections = detector.detect(image)

# Crop riders and detect helmets/seatbelts based on vehicle association
rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON] if 'EntityClass' in globals() else []
# Let's import EntityClass
from config.settings import EntityClass, TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
from core.scene_graph import is_four_wheeler_occupant

rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]

helmet_crops = []
seatbelt_crops = []
for r in rider_nodes:
    is_two_wheeler_rider = False
    is_occupant = False
    for v in vehicles:
        if v.entity_class in TWO_WHEELER_CLASSES:
            from core.scene_graph import compute_iou, compute_overlap_ratio
            if max(compute_iou(r.bbox, v.bbox), compute_overlap_ratio(r.bbox, v.bbox)) >= 0.15:
                is_two_wheeler_rider = True
        elif v.entity_class in FOUR_WHEELER_CLASSES:
            if is_four_wheeler_occupant(r.bbox, v.bbox):
                is_occupant = True
    if is_two_wheeler_rider:
        helmet_crops.append(r.bbox)
    if is_occupant:
        seatbelt_crops.append(r.bbox)
        
if helmet_crops:
    detections.extend(detector.detect_helmets(image, helmet_crops))
if seatbelt_crops:
    detections.extend(detector.detect_seatbelts(image, seatbelt_crops))
    
plate_detections = detector.detect_plates(image, detections)
detections.extend(plate_detections)

scene_graph_builder.build(detections, frame_shape=image.shape[:2])
violations = violation_engine.analyze(scene_graph_builder, image)

print("\n--- Violations Detected on wrong_side_sample.png ---")
for v in violations:
    print(f"Type: {v.violation_type.name}")
    print(f"Description: {v.description}")
    print(f"Involved: {v.involved_nodes}")
    print(f"Bbox: {v.bbox}")
    print(f"Metadata: {v.metadata}")
    print("-" * 30)
