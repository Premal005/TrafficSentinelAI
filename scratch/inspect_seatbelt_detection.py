import hashlib
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    image = cv2.imread(str(image_path))
    raw_md5 = hashlib.md5(image.tobytes()).hexdigest()
    print(f"Raw image.tobytes() MD5: {raw_md5}")
    
    # Let's also check if we encode it back to PNG
    _, encoded = cv2.imencode('.png', image)
    encoded_md5 = hashlib.md5(encoded.tobytes()).hexdigest()
    print(f"Encoded PNG MD5: {encoded_md5}")

if __name__ == "__main__":
    main()
