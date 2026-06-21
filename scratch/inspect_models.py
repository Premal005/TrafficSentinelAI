from ultralytics import YOLO
from pathlib import Path

weights_dir = Path("models/weights")
files = ["yolo11s_helmet.pt", "yolo11s_plate.pt", "yolov8s_helmet.pt", "custom_helmet_classifier.pth"]

for f in files:
    path = weights_dir / f
    if not path.exists():
        print(f"{f} does not exist")
        continue
    if f.endswith(".pt"):
        try:
            model = YOLO(str(path))
            print(f"{f}: Loaded successfully as YOLO model.")
            print(f"  Names: {model.names}")
            print(f"  Task: {model.task}")
            print(f"  Type: {type(model)}")
        except Exception as e:
            print(f"{f}: Failed to load as YOLO model: {e}")
    else:
        print(f"{f}: PyTorch classification model (size: {path.stat().st_size} bytes)")
