import torch
from kraken import blla
from PIL import Image
import numpy as np
import cv2
import os

IMAGE_PATH = r"C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR\flask_app_updated\uploads\page_0009_11945239_1779110413.jpg"

print(f"Testing Kraken segmentation on: {os.path.basename(IMAGE_PATH)}")

pil_image = Image.open(IMAGE_PATH).convert('RGB')
print(f"Image size: {pil_image.size}")

print("Running Kraken segmentation...")
result = blla.segment(pil_image)

print(f"Lines found: {len(result.lines)}")
print(f"Regions found: {len(result.regions)}")

# Inspect the first line object to see what attributes are available
first_line = result.lines[0]
print(f"Line object type: {type(first_line)}")
print(f"Line attributes: {[a for a in dir(first_line) if not a.startswith('_')]}")

# Draw visualisation using baseline points
img_array = np.array(pil_image)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

boxes = []
for i, line in enumerate(result.lines):
    # Kraken 7.x uses baseline points and boundary polygons
    # Try different attribute names
    if hasattr(line, 'baseline'):
        baseline = line.baseline
        if baseline and len(baseline) > 0:
            pts = np.array(baseline, dtype=np.int32)
            cv2.polylines(img_bgr, [pts], False, (0, 255, 0), 2)
            cv2.putText(img_bgr, str(i), (pts[0][0], pts[0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if hasattr(line, 'boundary'):
        boundary = line.boundary
        if boundary and len(boundary) > 0:
            pts = np.array(boundary, dtype=np.int32)
            # Get bounding box from boundary polygon
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)

print(f"Boxes extracted: {len(boxes)}")
if boxes:
    print(f"First 5 boxes: {boxes[:5]}")

output_path = "kraken_segmentation_test.jpg"
cv2.imwrite(output_path, img_bgr)
print(f"Visualisation saved to: {output_path}")
print("Green lines = baselines, Blue boxes = bounding boxes from boundary polygons")