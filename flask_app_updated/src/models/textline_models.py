"""
Textline detection models for Flask OCR Application
"""

import logging
import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from PIL import Image

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Detectron2 not available - textline detection will use fallback")

from ..exceptions import TextlineDetectionException, ModelLoadException
from ..config import ModelConfig
from .ocr_models import OCRModel

logger = logging.getLogger(__name__)


class TextlineExtractor:
    """Advanced textline extraction using Detectron2"""
    
    def __init__(self, config: ModelConfig, ocr_model: Optional[OCRModel] = None):
        self.config = config
        self.ocr_model = ocr_model
        self.predictor: Optional[DefaultPredictor] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_loaded = False
        
        # Initialize OCR model if not provided
        if self.ocr_model is None:
            self.ocr_model = OCRModel(config)
    
    def _setup_cfg(self, model_path: str):
        """Setup Detectron2 configuration"""
        if not DETECTRON2_AVAILABLE:
            raise ModelLoadException("Detectron2 not available")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # textline, baseline
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ("page_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        MetadataCatalog.get("page_test").thing_classes = ["textline", "baseline"]
        return cfg
    
    def load_model(self) -> None:
        """Load textline detection model"""
        if self._is_loaded:
            logger.info("Textline model already loaded")
            return
        
        if not DETECTRON2_AVAILABLE:
            logger.warning("Detectron2 not available - using fallback textline detection")
            self._is_loaded = True
            return
        
        if not self.config.textline_model_path:
            logger.warning("No textline model path provided - using fallback")
            self._is_loaded = True
            return
        
        if not os.path.exists(self.config.textline_model_path):
            logger.warning(f"Textline model path not found: {self.config.textline_model_path}")
            self._is_loaded = True
            return
        
        try:
            logger.info("Loading textline detection model...")
            cfg = self._setup_cfg(self.config.textline_model_path)
            self.predictor = DefaultPredictor(cfg)
            self._is_loaded = True
            logger.info("Textline detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load textline model: {e}")
            self._is_loaded = True  # Mark as loaded to use fallback
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    def health_check(self) -> dict:
        """Perform health check on the model"""
        return {
            'loaded': self.is_loaded(),
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_path': self.config.textline_model_path,
            'predictor_available': self.predictor is not None
        }
    
    def calculate_dynamic_padding(self, boxes: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, int]:
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
    
    def filter_margin_boxes_by_area(self, boxes: np.ndarray, scores: np.ndarray, 
                                  area_threshold_percent: float = 12.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    def detect_columns_and_sort_reading_order(self, boxes: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
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
    
    def extract_textlines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Extract textline predictions from image"""
        if not self.is_loaded():
            self.load_model()
        
        if self.predictor is None:
            # Use fallback method
            return self._fallback_textline_detection(image)
        
        try:
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Filter for textline class (assuming class 0 is textline)
            textline_mask = instances.pred_classes == 0
            textline_boxes = instances.pred_boxes[textline_mask].tensor.numpy()
            textline_scores = instances.scores[textline_mask].numpy()
            
            return textline_boxes, textline_scores, outputs
            
        except Exception as e:
            logger.error(f"Textline detection failed: {e}")
            return self._fallback_textline_detection(image)
    
    def _fallback_textline_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Fallback textline detection when Detectron2 is not available"""
        logger.info("Using fallback textline detection")
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
        
        return np.array(mock_boxes), np.array(mock_scores), {}
    
    def crop_textlines_with_dynamic_padding(self, image: np.ndarray, boxes: np.ndarray, 
                                          use_margin_filtering: bool = True) -> Tuple[List[np.ndarray], List[List[int]], Dict[str, int]]:
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
    
    def process_textlines_with_trocr(self, cropped_textlines: List[np.ndarray], 
                                   reading_order_info: List[Dict]) -> List[Dict]:
        """Process cropped textlines with TrOCR in sequential reading order"""
        logger.info(f"Processing {len(cropped_textlines)} textlines with TrOCR...")
        
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
                text = self.ocr_model.process_image(pil_image)
                
                # Store result with reading order information
                ocr_result = {
                    'reading_order_index': textline_data['reading_order_index'],
                    'column': textline_data['column'],
                    'position_in_column': textline_data['position_in_column'],
                    'original_index': textline_data['original_index'],
                    'text': text,
                    'confidence': 1.0
                }
                
                ocr_results.append(ocr_result)
                    
            except Exception as e:
                logger.error(f"Error processing textline {textline_data['reading_order_index']}: {str(e)}")
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
