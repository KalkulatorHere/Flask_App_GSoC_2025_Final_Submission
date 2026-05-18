import torch
import cv2
import numpy as np
import os
import sys

# Set up paths correctly before any imports
PROJECT_ROOT = r"C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR"
APP_DIR = os.path.join(PROJECT_ROOT, "flask_app_updated")

# Change to app directory and add to path
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from modular_app.config import get_config, setup_directories
from modular_app.ocr_models import OCRModelsService

print("=" * 60)
print("FULL PIPELINE TEST")
print("=" * 60)

IMAGE_PATH = os.path.join(APP_DIR, "uploads", "page_0009_11945239_1779110413.jpg")
print(f"Image exists: {os.path.exists(IMAGE_PATH)}")

config = get_config()

# Make paths absolute
base_dir = APP_DIR
config.upload_folder = os.path.join(base_dir, config.upload_folder)
config.annotation_folder = os.path.join(base_dir, config.annotation_folder)
config.inference_folder = os.path.join(base_dir, config.inference_folder)
config.line_segments_folder = os.path.join(base_dir, config.line_segments_folder)
setup_directories(config)

print("Loading TrOCR model...")
ocr_service = OCRModelsService(config)
ocr_service.load_trocr_model()
print("TrOCR loaded")

print(f"Testing segmentation on: {os.path.basename(IMAGE_PATH)}")
boxes, scores, image = ocr_service.extract_textlines_from_image(IMAGE_PATH)
print(f"Lines found: {len(boxes)}")

if len(boxes) > 0:
    print("Cropping line segments...")
    crops, padded_boxes = ocr_service.crop_textlines_with_padding(image, boxes, padding=5)
    print(f"Crops created: {len(crops)}")

    print("Running TrOCR on first 5 lines...")
    results = ocr_service.process_textlines_with_trocr(crops[:5])

    print("=" * 60)
    print("OCR RESULTS")
    print("=" * 60)
    for r in results:
        print(f"Line {r['line_index']}: {r['text']}")
else:
    print("No lines found - check Kraken installation")

print("=" * 60)
print("Pipeline test complete")