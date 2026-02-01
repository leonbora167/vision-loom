import os 
from PIL import Image 
import numpy as np 
import cv2

# Creating a generator so I can save storage in the RAM 
def load_images(path): #Either single file or folder 
    if os.path.isfile(path):
        image = Image.open(path)
        yield path, image 

    else:
        for file_name in os.listdir(path): #Assuming this is the string of the folder containing images 
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(path, file_name)
                image = Image.open(full_path)
                yield full_path, image 

def mask_to_polygon(mask: np.ndarray, epsilon_ratio=0.005): #Convert binary mask to polygon using OpenCV contours
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Polygon simplification
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)

    return polygon.squeeze(1)  # (N, 2)

def mask_to_contour_format(masks, image):
    array_img = np.array(image)
    image_h, image_w, _ = array_img.shape
    lines = []
    masks = masks.cpu().numpy()

    for i in range(masks.shape[0]):
        # Pick best mask (largest area)
        mask_set = masks[i]
        areas = [m.sum() for m in mask_set]
        best_mask = mask_set[np.argmax(areas)]

        polygon = mask_to_polygon(best_mask)
        if polygon is None or len(polygon) < 3:
            continue

        # Normalize coordinates
        polygon = polygon.astype(float)
        polygon[:, 0] /= image_w
        polygon[:, 1] /= image_h

        polygon = polygon.clip(0, 1)

        # Flatten
        poly_flat = polygon.reshape(-1)
        poly_str = " ".join(f"{p:.6f}" for p in poly_flat)

        #line = f"{class_ids[i]} {poly_str}"
        #lines.append(line)
        return poly_str