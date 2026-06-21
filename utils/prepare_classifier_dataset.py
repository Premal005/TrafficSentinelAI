"""
Prepare Classifier Dataset — TrafficSentinel AI.

Parses the full traffic scene images in 'Training data' and 'validation data'
folders, detects riders on two-wheelers using YOLO, crops their head regions,
and organizes them into a structured dataset for training a binary classifier.
"""

import os
import sys
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass
from core.entity_detector import EntityDetector
from core.scene_graph import SceneGraph, RIDES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def crop_rider_heads():
    # Source dataset
    dataset_dir = PROJECT_ROOT / "Traffic Violations Analysis Dataset"
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found at {dataset_dir}")
        return

    # Target directory
    target_dir = PROJECT_ROOT / "data" / "classifier_dataset"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing entity detector for dataset preparation...")
    detector = EntityDetector(SETTINGS)

    # Process splits
    splits = [
        ("Training data", "train"),
        ("validation data", "val")
    ]

    categories = ["helmet", "no_helmet"]

    for src_split, dest_split in splits:
        logger.info(f"Processing split: {src_split} -> {dest_split}")
        
        for category in categories:
            src_cat_dir = dataset_dir / src_split / category
            dest_cat_dir = target_dir / dest_split / category
            dest_cat_dir.mkdir(parents=True, exist_ok=True)

            if not src_cat_dir.exists():
                logger.warning(f"Source folder not found: {src_cat_dir}")
                continue

            images = list(src_cat_dir.glob("*.jpg")) + list(src_cat_dir.glob("*.png"))
            limit = 100 if dest_split == "train" else 30
            images = images[:limit]
            logger.info(f"Found {len(images)} images in {src_cat_dir.name} (capped at {limit})")

            crop_count = 0
            for img_path in tqdm(images, desc=f"Cropping {dest_split}/{category}"):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                h, w = frame.shape[:2]

                # Run direct entity detector (person/vehicles)
                detections = detector.detect(frame, use_sahi=False)

                # Build scene graph to associate riders with vehicles
                sg = SceneGraph(SETTINGS)
                sg.build(detections)

                # Query person RIDES motorcycle/bicycle
                # Any rider detected in a 'helmet' folder is assumed to be 'helmet' target,
                # and any rider in 'no_helmet' is assumed to be 'no_helmet' target.
                riders = []
                for node_id, node in sg.nodes.items():
                    if node.entity_class in [EntityClass.PERSON, EntityClass.RIDER]:
                        # Check if riding a two-wheeler
                        vehicle = sg.get_vehicle_of(node_id)
                        if vehicle and vehicle.entity_class in [EntityClass.MOTORCYCLE, EntityClass.BICYCLE]:
                            riders.append(node)

                # Fallback: if no associated riders found, take all persons in the scene
                if not riders:
                    riders = [n for n in sg.nodes.values() if n.entity_class in [EntityClass.PERSON, EntityClass.RIDER]]

                for idx, rider in enumerate(riders):
                    rx1, ry1, rx2, ry2 = rider.bbox
                    rider_h = ry2 - ry1

                    # Expand crop upward by 35% to fully capture head/helmet
                    y1_expanded = max(0, ry1 - int(rider_h * 0.35))
                    
                    # We only need the top 50% of the expanded rider bbox (head + shoulders)
                    head_h = (ry2 - y1_expanded) // 2
                    y2_head = y1_expanded + head_h

                    head_crop = frame[y1_expanded:y2_head, rx1:rx2]
                    
                    if head_crop.size == 0 or head_crop.shape[0] < 16 or head_crop.shape[1] < 16:
                        continue

                    # Save crop
                    dest_filename = f"{img_path.stem}_rider_{idx}.jpg"
                    dest_path = dest_cat_dir / dest_filename
                    cv2.imwrite(str(dest_path), head_crop)
                    crop_count += 1

            logger.info(f"Saved {crop_count} head crops to {dest_cat_dir}")


if __name__ == "__main__":
    crop_rider_heads()
