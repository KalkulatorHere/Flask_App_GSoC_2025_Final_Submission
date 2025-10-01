"""
Image processing utilities for Flask OCR Application
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility class for image processing operations"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Save image to file path"""
        try:
            success = cv2.imwrite(output_path, image)
            if not success:
                logger.error(f"Failed to save image: {output_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {e}")
            return False
    
    @staticmethod
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        """Get image dimensions (width, height)"""
        if image is None:
            return (0, 0)
        height, width = image.shape[:2]
        return (width, height)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Resize image to target dimensions"""
        try:
            resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def resize_image_by_factor(image: np.ndarray, factor: float) -> np.ndarray:
        """Resize image by scaling factor"""
        try:
            height, width = image.shape[:2]
            new_width = int(width * factor)
            new_height = int(height * factor)
            return ImageUtils.resize_image(image, new_width, new_height)
        except Exception as e:
            logger.error(f"Error resizing image by factor: {e}")
            return image
    
    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Crop image to specified region"""
        try:
            return image[y:y+height, x:x+width]
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            return image
    
    @staticmethod
    def split_image_horizontally(image: np.ndarray, split_point: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into left and right halves"""
        try:
            height, width = image.shape[:2]
            if split_point is None:
                split_point = width // 2
            
            left_half = image[:, :split_point]
            right_half = image[:, split_point:]
            
            return left_half, right_half
        except Exception as e:
            logger.error(f"Error splitting image: {e}")
            return image, image
    
    @staticmethod
    def split_image_vertically(image: np.ndarray, split_point: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into top and bottom halves"""
        try:
            height, width = image.shape[:2]
            if split_point is None:
                split_point = height // 2
            
            top_half = image[:split_point, :]
            bottom_half = image[split_point:, :]
            
            return top_half, bottom_half
        except Exception as e:
            logger.error(f"Error splitting image vertically: {e}")
            return image, image
    
    @staticmethod
    def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB"""
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error converting BGR to RGB: {e}")
            return image
    
    @staticmethod
    def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR"""
        try:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error converting RGB to BGR: {e}")
            return image
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        try:
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error converting PIL to CV2: {e}")
            return np.array(pil_image)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        try:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        except Exception as e:
            logger.error(f"Error converting CV2 to PIL: {e}")
            return Image.fromarray(cv2_image)
    
    @staticmethod
    def enhance_image_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 30) -> np.ndarray:
        """Enhance image contrast"""
        try:
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing image contrast: {e}")
            return image
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Apply denoising to image"""
        try:
            denoised = cv2.fastNlMeansDenoising(image)
            return denoised
        except Exception as e:
            logger.error(f"Error denoising image: {e}")
            return image
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Apply Gaussian blur to image"""
        try:
            blurred = cv2.GaussianBlur(image, kernel_size, 0)
            return blurred
        except Exception as e:
            logger.error(f"Error applying Gaussian blur: {e}")
            return image
    
    @staticmethod
    def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Detect edges in image using Canny edge detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            return edges
        except Exception as e:
            logger.error(f"Error detecting edges: {e}")
            return image
    
    @staticmethod
    def find_contours(image: np.ndarray) -> List[np.ndarray]:
        """Find contours in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            logger.error(f"Error finding contours: {e}")
            return []
    
    @staticmethod
    def draw_rectangles(image: np.ndarray, rectangles: List[Tuple[int, int, int, int]], 
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw rectangles on image"""
        try:
            result = image.copy()
            for x, y, w, h in rectangles:
                cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            return result
        except Exception as e:
            logger.error(f"Error drawing rectangles: {e}")
            return image
    
    @staticmethod
    def validate_image_format(image: np.ndarray) -> bool:
        """Validate that image is in correct format for processing"""
        try:
            if image is None:
                return False
            
            if len(image.shape) not in [2, 3]:
                return False
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating image format: {e}")
            return False
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Get comprehensive image information"""
        try:
            if image is None:
                return {
                    'valid': False,
                    'error': 'Image is None'
                }
            
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            dtype = str(image.dtype)
            
            return {
                'valid': True,
                'width': width,
                'height': height,
                'channels': channels,
                'dtype': dtype,
                'total_pixels': width * height,
                'memory_size': image.nbytes
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
