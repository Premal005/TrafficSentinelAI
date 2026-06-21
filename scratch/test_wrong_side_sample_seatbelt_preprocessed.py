import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from core.entity_detector import EntityDetector
from core.scene_conditioner import SceneConditioner
from core.scene_graph import is_four_wheeler_occupant
from config.settings import SETTINGS, EntityClass, TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES

detector = EntityDetector()
conditioner = SceneConditioner(SETTINGS)
image_path = 'data/sample_images/wrong_side_sample.png'
image = cv2.imread(image_path)

# Run SceneConditioner (Layer 1)
cond_res = conditioner.condition(image)
enhanced_frame = cond_res.enhanced_image
deg_type = cond_res.degradation_type.value
quality_score = cond_res.quality_score

print(f"Conditioner details: degradation={deg_type}, quality_score={quality_score:.3f}")

detections = detector.detect(enhanced_frame)
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
    seatbelt_detections = detector.detect_seatbelts(enhanced_frame, seatbelt_crops)
    print("\nSeatbelt detections generated on preprocessed image:")
    for d in seatbelt_detections:
        print(f"  {d.class_name} @ {d.bbox} conf={d.confidence:.2f}")
else:
    print("No seatbelt crops found.")
