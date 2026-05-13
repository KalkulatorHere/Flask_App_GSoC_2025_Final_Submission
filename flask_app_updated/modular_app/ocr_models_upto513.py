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

class TextlineExtractor:
    """Advanced textline extraction class"""
    
    def __init__(self, model_path: str):
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 not available")
            
        self.cfg = self.setup_cfg(model_path)
        self.predictor = DefaultPredictor(self.cfg)
        
        # --- NEW TrOCR LOCAL SETUP ---
        print("Loading TrOCR model in TextlineExtractor...")
        
        # Path to your downloaded model
        SPANISH_PATH = r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"
        
        if TRANSFORMERS_AVAILABLE:
            # Determine device first so we can move the model immediately
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                print(f"Loading local Spanish model from: {SPANISH_PATH}")
                self.trocr_processor = TrOCRProcessor.from_pretrained(SPANISH_PATH)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    SPANISH_PATH,
                    low_cpu_mem_usage=False,  # FIX: Prevents meta device error
                    local_files_only=True     # Ensure it doesn't try to use the web
                ).to(self.device)             # Move to CPU/GPU immediately
                
                print(f"TrOCR Spanish model loaded on {self.device}")

            except Exception as e:
                print(f"Local Spanish model failed: {e}")
                print("Using cloud fallback (Internet required)...")
                # Fallback to the cloud version if the local folder is empty/broken
                self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    'microsoft/trocr-base-printed'
                ).to(self.device)
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
        
        # Calculate centers of bounding boxes
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate vertical and horizontal distances
        vertical_distances = []
        horizontal_distances = []
        
        # Sort boxes by y-coordinate to find vertical neighbors
        sorted_indices = np.argsort(centers[:, 1])
        sorted_boxes = boxes[sorted_indices]
        
        # Calculate vertical gaps between consecutive textlines
        for i in range(len(sorted_boxes) - 1):
            current_box = sorted_boxes[i]
            next_box = sorted_boxes[i + 1]
            
            # Check if boxes are roughly horizontally aligned
            current_center_x = (current_box[0] + current_box[2]) / 2
            next_center_x = (next_box[0] + next_box[2]) / 2
            
            if abs(current_center_x - next_center_x) < image_shape[1] * 0.3:
                gap = next_box[1] - current_box[3]
                if gap > 0:
                    vertical_distances.append(gap)
        
        # Calculate average distances
        avg_vertical_gap = np.median(vertical_distances) if vertical_distances else 20
        avg_horizontal_gap = 15  # Default for horizontal
        
        # Calculate dynamic padding
        vertical_padding = max(5, min(25, avg_vertical_gap / 2))
        horizontal_padding = max(3, min(20, avg_horizontal_gap / 3))
        
        # Consider box heights
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
        
        # Calculate areas
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        areas = np.array(areas)
        avg_area = np.mean(areas)
        area_threshold = avg_area * (area_threshold_percent / 100.0)
        
        # Filter boxes
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
        
        # Calculate centers
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Single column - sort by y-coordinate only
        all_indices = np.arange(len(boxes))
        y_coords = centers[:, 1]
        y_sort_indices = np.argsort(y_coords)
        
        # Create sorted results
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
        
        # Filter for textline class (assuming class 0 is textline)
        textline_mask = instances.pred_classes == 0
        textline_boxes = instances.pred_boxes[textline_mask].tensor.numpy()
        textline_scores = instances.scores[textline_mask].numpy()
        
        return textline_boxes, textline_scores, outputs
        
    def crop_textlines_with_dynamic_padding(self, image, boxes, use_margin_filtering=True):
        """Crop textline regions from image with dynamic padding"""
        if len(boxes) == 0:
            return [], [], {}
        
        # Filter by area if needed
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
            
            # Apply dynamic padding
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
        
        # Create list of textlines with their reading order
        textlines_with_order = []
        for i, (textline_crop, ro_info) in enumerate(zip(cropped_textlines, reading_order_info)):
            textlines_with_order.append({
                'crop': textline_crop,
                'reading_order_index': ro_info['reading_order_index'],
                'column': ro_info['column'],
                'position_in_column': ro_info['position_in_column'],
                'original_index': ro_info['original_index']
            })
        
        # Sort by reading order index
        textlines_with_order.sort(key=lambda x: x['reading_order_index'])
        
        # Process textlines in sequential order
        ocr_results = []
        
        for idx, textline_data in enumerate(textlines_with_order):
            try:
                # Convert OpenCV image to PIL Image
                crop_bgr = textline_data['crop']
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                
                # Process with TrOCR
                pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text
                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(pixel_values, max_new_tokens=128)
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Store result with reading order information
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
        
        # Ensure results are sorted by reading order
        ocr_results.sort(key=lambda x: x['reading_order_index'])
        return ocr_results

class OCRModelsService:
    """Service for managing OCR models and processing"""
    
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.textline_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Add this line
        
    def load_trocr_model(self):
    #Load TrOCR model and processor
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
            
        if self.processor is None or self.model is None:
            print("Loading TrOCR model...")
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Local model paths
            SPANISH_MODEL_PATH = os.environ.get(
                'TROCR_SPANISH_MODEL_PATH',
                r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"
            )
            FALLBACK_MODEL_PATH = os.environ.get(
                'TROCR_FALLBACK_MODEL_PATH',
                r"C:\Users\rdb104\Documents\caserepos\models\ms_trocr"
            )
            #SPANISH_MODEL_PATH = r"C:\Users\rdb104\Documents\caserepos\models\trocr_span"
            #FALLBACK_MODEL_PATH = r"C:\Users\rdb104\Documents\caserepos\models\ms_trocr"
            
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
                    torch_dtype=torch.float32,      # Force full precision load
                    ignore_mismatched_sizes=True    # Handle the embed_positions mismatch
                )
                self.model = self.model.to(self.device)
                self.model.eval()                   # Set to evaluation mode
                
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
                    ).to(self.device)
                    self.model = self.model.to(self.device)

                    # Force all weights to materialise on the device
                    for param in self.model.parameters():
                        if param.device.type == 'meta':
                            raise RuntimeError(f"Parameter still on meta device after loading!")
                            
                    self.model.eval()
                    print(f"TrOCR Spanish model loaded successfully on {self.device}!")
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
        """Extract textlines from image using advanced pipeline or fallback"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Try advanced pipeline first
            if self.textline_extractor is not None:
                try:
                    boxes, scores, outputs = self.textline_extractor.extract_textlines(image)
                    return boxes, scores, image
                except Exception as e:
                    print(f"Advanced textline detection failed: {e}")
            
            # Fallback to mock data if advanced pipeline fails
            print("Using fallback mock textline detection")
            height, width = image.shape[:2]
            mock_boxes = []
            mock_scores = []
            
            # Create some mock textlines
            line_height = height // 10
            for i in range(5):
                y1 = i * line_height + 50
                y2 = (i + 1) * line_height - 20
                x1 = 50
                x2 = width - 50
                mock_boxes.append([x1, y1, x2, y2])
                mock_scores.append(0.9)
            
            return np.array(mock_boxes), np.array(mock_scores), image
            
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
            
            # Apply padding
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
                # Convert OpenCV image to PIL Image
                crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                
                # Process with TrOCR
                device = next(self.model.parameters()).device
                pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
                
                # Generate text
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
            # Add this debug line
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
