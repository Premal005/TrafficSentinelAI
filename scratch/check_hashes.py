import hashlib
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).parent.parent

def get_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    for p in sample_dir.glob("*.png"):
        img = cv2.imread(str(p))
        shape = img.shape if img is not None else "Error"
        print(f"File: {p.name}")
        print(f"  Shape: {shape}")
        print(f"  MD5: {get_md5(p)}")

if __name__ == "__main__":
    main()
