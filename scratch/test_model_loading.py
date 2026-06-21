from ultralytics import YOLO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

def test_load(filename):
    path = PROJECT_ROOT / "models" / "weights" / filename
    if not path.exists():
        print(f"{filename}: DOES NOT EXIST")
        return
    try:
        model = YOLO(str(path))
        print(f"{filename}: Loaded OK. Classes: {model.names}")
    except Exception as e:
        print(f"{filename}: FAILED to load: {e}")

def main():
    print("Testing weights loading...")
    test_load("yolo11s_helmet.pt")
    test_load("yolov8s_helmet.pt")
    test_load("yolo11s_plate.pt")
    test_load("yolov8s_plate.pt")

if __name__ == "__main__":
    main()
