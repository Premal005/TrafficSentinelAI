import cv2
import easyocr
import time
import os

img_path = r"C:\Users\hp\.gemini\antigravity-ide\brain\a931e15a-e458-48ce-b68c-11b2929c9f25\illegal_parking_sample_1781982036473.png"
if not os.path.exists(img_path):
    print(f"Error: image not found at {img_path}")
    exit()

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Loaded image {img_path} with size {w}x{h}")

# Downscale for faster OCR
scale = 640.0 / max(h, w)
resized = cv2.resize(img, (int(w * scale), int(h * scale)))
print(f"Resized image to {resized.shape[1]}x{resized.shape[0]}")

t0 = time.time()
reader = easyocr.Reader(["en"], gpu=False)  # Let's test with CPU first
t1 = time.time()
print(f"Initialized EasyOCR in {t1 - t0:.2f}s")

t2 = time.time()
results = reader.readtext(resized)
t3 = time.time()
print(f"Ran OCR on resized image in {t3 - t2:.2f}s")

for bbox, text, conf in results:
    # Scale back coordinates
    bx1 = int(bbox[0][0] / scale)
    by1 = int(bbox[0][1] / scale)
    bx2 = int(bbox[2][0] / scale)
    by2 = int(bbox[2][1] / scale)
    print(f"Detected: '{text}' (conf={conf:.2f}) at bbox=({bx1}, {by1}, {bx2}, {by2})")
