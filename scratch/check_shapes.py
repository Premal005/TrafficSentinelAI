import cv2
img1 = cv2.imread('data/sample_images/wrong_side_sample.png')
img2 = cv2.imread('data/sample_images/2. seatbelt.png')
print(f"wrong_side_sample.png shape: {img1.shape if img1 is not None else 'None'}")
print(f"2. seatbelt.png shape: {img2.shape if img2 is not None else 'None'}")
