"""
OCR Models Service - Handles model loading and OCR processing
Includes TrOCR and Detectron2 textline detection models
"""
import os
import logging
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

# Optional imports with guards
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available")

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logging.warning("detectron2 not available")

logger = logging.getLogger(__name__)


def _fix_meta_tensors(model, device):
    """
    Fix meta-device tensors that survive from_pretrained() and cause
    'Tensor on device meta is not on the expected device cuda:0' at inference.

    Root cause (confirmed by inspection):
      TrOCRSinusoidalPositionalEmbedding stores its sinusoidal weight table as
      a plain Python attribute self.weights (NOT a buffer or parameter), so
      model.to(device) never moves it. The checkpoint was saved with an older
      transformers version that stored it as a buffer named _float_tensor;
      current transformers loads it into self.weights as a meta tensor and
      reports it as UNEXPECTED in the load report.

    Fix: recompute self.weights directly on the target device and re-register
    it as a proper buffer so it stays on GPU and .to(device) keeps it there.
    """
    # Fix parameters (safety net)
    for name, param in list(model.named_parameters()):
        if param.device.type == 'meta':
            logger.warning(f"Parameter '{name}' still on meta device - replacing with zeros on {device}")
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(
                parent,
                parts[-1],
                torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device=device))
            )

    # Fix ALL buffers including non-persistent ones (_buffers dict, not named_buffers())
    for module_name, module in model.named_modules():
        for buf_name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type == 'meta':
                full_name = f"{module_name}.{buf_name}" if module_name else buf_name
                logger.warning(f"Buffer '{full_name}' still on meta device - replacing on {device}")
                module._buffers[buf_name] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)

    # PRIMARY FIX: TrOCRSinusoidalPositionalEmbedding.weights is a plain Python
    # attribute invisible to .to() and named_buffers(). Recompute it directly on
    # the target device and re-register as a proper buffer so it stays on GPU.
    for module_name, module in model.named_modules():
        if hasattr(module, 'weights') and isinstance(module.weights, torch.Tensor):
            if module.weights.device.type == 'meta':
                num_embeddings = module.weights.shape[0]
                embedding_dim  = module.weights.shape[1]
                padding_idx    = getattr(module, 'padding_idx', None)
                # get_embedding() always creates on CPU; move immediately to target device
                real_weights = module.get_embedding(
                    num_embeddings, embedding_dim, padding_idx
                ).to(device)
                # Must delete the plain Python attribute first — register_buffer
                # raises "attribute already exists" if the name is taken by __dict__
                if 'weights' in module.__dict__:
                    del module.__dict__['weights']
                # Register as a proper buffer so future .to() calls keep it on device
                module.register_buffer('weights', real_weights, persistent=False)
                logger.warning(
                    f"Re-registered '{module_name}.weights' directly on {device} "
                    f"(shape={real_weights.shape})"
                )

    return model


class TextlineExtractor:
    """Advanced textline extraction class"""

    def __init__(self, model_path: str):
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 not available")

        self.cfg = self.setup_cfg(model_path)
        self.predictor = DefaultPredictor(self.cfg)

        # --- TrOCR LOCAL SETUP ---
        print("Loading TrOCR model in TextlineExtractor...")

        SPANISH_PATH = r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"

        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            try:
                print(f"Loading local Spanish model from: {SPANISH_PATH}")
                self.trocr_processor = TrOCRProcessor.from_pretrained(SPANISH_PATH)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    SPANISH_PATH,
                    low_cpu_mem_usage=False,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True
                )

                # Fix meta-device tensors, placing weights directly on GPU
                self.trocr_model = _fix_meta_tensors(self.trocr_model, self.device)
                self.trocr_model = self.trocr_model.to(self.device)
                self.trocr_model.eval()

                print(f"TrOCR Spanish model loaded on {self.device}")

            except Exception as e:
                print(f"Local Spanish model failed: {e}")
                print("Using cloud fallback (Internet required)...")
                self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    'microsoft/trocr-base-printed'
                )
                self.trocr_model = _fix_meta_tensors(self.trocr_model, self.device)
                self.trocr_model = self.trocr_model.to(self.device)
                self.trocr_model.eval()
                print(f"Fallback model loaded on {self.device}")
        else:
            raise ImportError("Transformers not available")

    def setup_cfg(self, model_path: str):
        """Setup Detectron2 configuration"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # textline, baseline
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ("page_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        MetadataCatalog.get("page_test").thing_classes = ["textline", "baseline"]
        return cfg

    def calculate_dynamic_padding(self, boxes, image_shape):
        """Calculate dynamic padding based on average distances between textboxes"""
        if len(boxes) < 2:
            return {"top": 10, "bottom": 10, "left": 8, "right": 8}

        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])

        centers = np.array(centers)

        vertical_distances = []
        horizontal_distances = []

        sorted_indices = np.argsort(centers[:, 1])
        sorted_boxes = boxes[sorted_indices]

        for i in range(len(sorted_boxes) - 1):
            current_box = sorted_boxes[i]
            next_box = sorted_boxes[i + 1]

            current_center_x = (current_box[0] + current_box[2]) / 2
            next_center_x = (next_box[0] + next_box[2]) / 2

            if abs(current_center_x - next_center_x) < image_shape[1] * 0.3:
                gap = next_box[1] - current_box[3]
                if gap > 0:
                    vertical_distances.append(gap)

        avg_vertical_gap = np.median(vertical_distances) if vertical_distances else 20
        avg_horizontal_gap = 15

        vertical_padding = max(5, min(25, avg_vertical_gap / 2))
        horizontal_padding = max(3, min(20, avg_horizontal_gap / 3))

        box_heights = [box[3] - box[1] for box in boxes]
        avg_height = np.mean(box_heights)
        height_factor = max(0.1, min(0.3, avg_height / 100))
        vertical_padding = max(vertical_padding, avg_height * height_factor)

        padding = {
            "top": int(vertical_padding * 0.95),
            "bottom": int(vertical_padding * 1.2),
            "left": int(horizontal_padding),
            "right": int(horizontal_padding)
        }

        return padding

    def filter_margin_boxes_by_area(self, boxes, scores, area_threshold_percent=12.5):
        """Filter out boxes that are significantly smaller than average"""
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)

        areas = np.array(areas)
        avg_area = np.mean(areas)
        area_threshold = avg_area * (area_threshold_percent / 100.0)

        main_boxes = []
        main_scores = []
        margin_boxes = []
        margin_scores = []

        for box, score, area in zip(boxes, scores, areas):
            if area >= area_threshold:
                main_boxes.append(box)
                main_scores.append(score)
            else:
                margin_boxes.append(box)
                margin_scores.append(score)

        return np.array(main_boxes), np.array(main_scores), np.array(margin_boxes), np.array(margin_scores)

    def detect_columns_and_sort_reading_order(self, boxes, scores):
        """Sort textlines in TOP to BOTTOM order (single column)"""
        if len(boxes) == 0:
            return boxes, scores, []

        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])

        centers = np.array(centers)

        all_indices = np.arange(len(boxes))
        y_coords = centers[:, 1]
        y_sort_indices = np.argsort(y_coords)

        sorted_boxes = []
        sorted_scores = []
        reading_order = []

        for position, sort_idx in enumerate(y_sort_indices):
            original_idx = sort_idx
            sorted_boxes.append(boxes[original_idx])
            sorted_scores.append(scores[original_idx])

            reading_order.append({
                'original_index': int(original_idx),
                'column': 0,
                'position_in_column': int(position),
                'reading_order_index': int(position)
            })

        return np.array(sorted_boxes), np.array(sorted_scores), reading_order

    def extract_textlines(self, image):
        """Extract textline predictions from image"""
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        textline_mask = instances.pred_classes == 0
        textline_boxes = instances.pred_boxes[textline_mask].tensor.numpy()
        textline_scores = instances.scores[textline_mask].numpy()

        return textline_boxes, textline_scores, outputs

    def crop_textlines_with_dynamic_padding(self, image, boxes, use_margin_filtering=True):
        """Crop textline regions from image with dynamic padding"""
        if len(boxes) == 0:
            return [], [], {}

        if use_margin_filtering:
            main_boxes, main_scores, margin_boxes, margin_scores = self.filter_margin_boxes_by_area(
                boxes, np.ones(len(boxes))
            )

            if len(main_boxes) > 0:
                padding = self.calculate_dynamic_padding(main_boxes, image.shape)
                boxes_for_cropping = main_boxes
            else:
                padding = self.calculate_dynamic_padding(boxes, image.shape)
                boxes_for_cropping = boxes
        else:
            padding = self.calculate_dynamic_padding(boxes, image.shape)
            boxes_for_cropping = boxes

        cropped_textlines = []
        padded_boxes = []

        for box in boxes_for_cropping:
            x1, y1, x2, y2 = box.astype(int)

            x1_padded = max(0, x1 - padding["left"])
            y1_padded = max(0, y1 - padding["top"])
            x2_padded = min(image.shape[1], x2 + padding["right"])
            y2_padded = min(image.shape[0], y2 + padding["bottom"])

            cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
            cropped_textlines.append(cropped)
            padded_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])

        return cropped_textlines, padded_boxes, padding

    def process_textlines_with_trocr(self, cropped_textlines, reading_order_info):
        """Process cropped textlines with TrOCR in sequential reading order"""
        print(f"Processing {len(cropped_textlines)} textlines with TrOCR...")

        textlines_with_order = []
        for i, (textline_crop, ro_info) in enumerate(zip(cropped_textlines, reading_order_info)):
            textlines_with_order.append({
                'crop': textline_crop,
                'reading_order_index': ro_info['reading_order_index'],
                'column': ro_info['column'],
                'position_in_column': ro_info['position_in_column'],
                'original_index': ro_info['original_index']
            })

        textlines_with_order.sort(key=lambda x: x['reading_order_index'])

        ocr_results = []

        for idx, textline_data in enumerate(textlines_with_order):
            try:
                crop_bgr = textline_data['crop']
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)

                pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(pixel_values, max_new_tokens=128)
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                ocr_result = {
                    'reading_order_index': textline_data['reading_order_index'],
                    'column': textline_data['column'],
                    'position_in_column': textline_data['position_in_column'],
                    'original_index': textline_data['original_index'],
                    'text': generated_text.strip(),
                    'confidence': 1.0
                }

                ocr_results.append(ocr_result)

            except Exception as e:
                print(f"Error processing textline {textline_data['reading_order_index']}: {str(e)}")
                ocr_results.append({
                    'reading_order_index': textline_data['reading_order_index'],
                    'column': textline_data['column'],
                    'position_in_column': textline_data['position_in_column'],
                    'original_index': textline_data['original_index'],
                    'text': '',
                    'confidence': 0.0
                })

        ocr_results.sort(key=lambda x: x['reading_order_index'])
        return ocr_results


class OCRModelsService:
    """Service for managing OCR models and processing"""

    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.textline_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_trocr_model(self):
        """Load TrOCR model and processor"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")

        if self.processor is None or self.model is None:
            print("Loading TrOCR model...")

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")

            SPANISH_MODEL_PATH = os.environ.get(
                'TROCR_SPANISH_MODEL_PATH',
                r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"
            )
            FALLBACK_MODEL_PATH = os.environ.get(
                'TROCR_FALLBACK_MODEL_PATH',
                r"C:\Users\rdb104\Documents\caserepos\models\ms_trocr"
            )

            try:
                print(f"Loading local Spanish TrOCR model from: {SPANISH_MODEL_PATH}")
                self.processor = TrOCRProcessor.from_pretrained(
                    SPANISH_MODEL_PATH,
                    local_files_only=True
                )
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    SPANISH_MODEL_PATH,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True
                )

                # Fix meta-device tensors, placing weights directly on GPU
                self.model = _fix_meta_tensors(self.model, self.device)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Spanish TrOCR model loaded successfully on {self.device}!")

            except Exception as e:
                print(f"Local Spanish model failed: {e}")
                print(f"Trying fallback model from: {FALLBACK_MODEL_PATH}")
                try:
                    self.processor = TrOCRProcessor.from_pretrained(
                        FALLBACK_MODEL_PATH,
                        local_files_only=True
                    )
                    self.model = VisionEncoderDecoderModel.from_pretrained(
                        FALLBACK_MODEL_PATH,
                        local_files_only=True,
                        low_cpu_mem_usage=False
                    )

                    # Fix meta-device tensors, placing weights directly on GPU
                    self.model = _fix_meta_tensors(self.model, self.device)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    print(f"Fallback TrOCR model loaded successfully on {self.device}!")

                except Exception as e2:
                    print(f"Local fallback model failed: {e2}")
                    raise e2

    def load_textline_model(self):
        """Load advanced Detectron2 textline detection model"""
        if not DETECTRON2_AVAILABLE:
            logger.warning("Detectron2 not available, textline extraction will use fallback")
            return

        if self.textline_extractor is None:
            print("Loading advanced textline detection model...")
            try:
                if not os.path.exists(self.config.textline_model_path):
                    print(f"Model path {self.config.textline_model_path} not found, using mock predictor")
                    self.textline_extractor = None
                    return

                self.textline_extractor = TextlineExtractor(self.config.textline_model_path)
                print("Advanced textline detection model loaded successfully!")
            except Exception as e:
                print(f"Error loading advanced textline model: {e}")
                self.textline_extractor = None

    def extract_textlines_from_image(self, image_path: str):
        """Extract textlines using Kraken baseline segmentation with fallback"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Try Kraken segmentation first
            try:
                from kraken import blla
                from PIL import Image as PILImage
                import numpy as np

                print("Running Kraken baseline segmentation...")
                pil_image = PILImage.open(image_path).convert('RGB')
                result = blla.segment(pil_image)

                boxes = []
                for line in result.lines:
                    if hasattr(line, 'boundary') and line.boundary:
                        pts = np.array(line.boundary, dtype=np.int32)
                        x1 = int(pts[:, 0].min())
                        y1 = int(pts[:, 1].min())
                        x2 = int(pts[:, 0].max())
                        y2 = int(pts[:, 1].max())
                        # Filter out boxes that are too small to be real lines
                        if (x2 - x1) > 50 and (y2 - y1) > 10:
                            boxes.append([x1, y1, x2, y2])

                if len(boxes) > 0:
                    print(f"Kraken found {len(boxes)} text lines")
                    return np.array(boxes), np.ones(len(boxes)), image

                print("Kraken found no lines, falling back to mock")

            except Exception as e:
                print(f"Kraken segmentation failed: {e}")

            # Fallback mock only if Kraken fails completely
            print("Using fallback mock textline detection")
            height, width = image.shape[:2]
            mock_boxes = []
            line_height = height // 10
            for i in range(5):
                y1 = i * line_height + 50
                y2 = (i + 1) * line_height - 20
                mock_boxes.append([50, y1, width - 50, y2])

            return np.array(mock_boxes), np.ones(len(mock_boxes)), image

        except Exception as e:
            print(f"Textline extraction error: {str(e)}")
            return np.array([]), np.array([]), None

    def crop_textlines_with_padding(self, image, boxes, padding=10):
        """Crop textline regions from image with padding"""
        if len(boxes) == 0:
            return [], []

        cropped_textlines = []
        padded_boxes = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]

            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(image.shape[1], x2 + padding)
            y2_padded = min(image.shape[0], y2 + padding)

            cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
            cropped_textlines.append(cropped)
            padded_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])

        return cropped_textlines, padded_boxes

    def process_textlines_with_trocr(self, cropped_textlines):
        """Process cropped textlines with TrOCR"""
        if not cropped_textlines:
            return []

        self.load_trocr_model()
        ocr_results = []

        for idx, textline_crop in enumerate(cropped_textlines):
            try:
                crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)

                device = next(self.model.parameters()).device
                pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values, max_new_tokens=128)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                ocr_results.append({
                    'line_index': idx,
                    'text': generated_text.strip(),
                    'confidence': 1.0
                })

            except Exception as e:
                print(f"Error processing textline {idx}: {str(e)}")
                ocr_results.append({
                    'line_index': idx,
                    'text': '',
                    'confidence': 0.0
                })

        return ocr_results

    def perform_ocr_inference(self, image_path: str) -> str:
        """Perform TrOCR inference on an image (fallback method)"""
        try:
            self.load_trocr_model()
            image = Image.open(image_path).convert("RGB")
            device = next(self.model.parameters()).device
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = self.model.generate(pixel_values, max_new_tokens=128)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return f"Error during OCR: {str(e)}"
