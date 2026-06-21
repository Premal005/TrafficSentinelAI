import sys
from pathlib import Path
from ultralytics import YOLO

weights_dir = Path("E:/flipkart2r/models/weights")
models = {
    "yolo11m (vehicle)": Path("E:/flipkart2r/yolo11m.pt"),
    "yolo11s_helmet": weights_dir / "yolo11s_helmet.pt",
    "yolo11s_plate": weights_dir / "yolo11s_plate.pt",
    "yolov8s_helmet": weights_dir / "yolov8s_helmet.pt"
}

for name, path in models.items():
    print(f"=== {name} ({path}) ===")
    if not path.exists():
        print("  File does not exist.")
        continue
    try:
        model = YOLO(str(path))
        print(f"  Task: {model.task}")
        print(f"  Names: {model.names}")
    except Exception as e:
        print(f"  Error: {e}")
