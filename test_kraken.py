import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

import importlib.metadata
print(f"Kraken: {importlib.metadata.version('kraken')}")

from kraken import blla
print("Kraken blla module imported successfully")

from PIL import Image
import numpy as np

test_image = Image.fromarray(np.ones((200, 800, 3), dtype=np.uint8) * 255)
result = blla.segment(test_image)
print(f"Segmentation successful")
print(f"Lines found on blank image: {len(result.lines)}")