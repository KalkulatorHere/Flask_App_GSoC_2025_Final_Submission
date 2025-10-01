"""
OCR model management for Flask OCR Application
"""

import logging
import torch
from typing import Optional, Tuple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..exceptions import ModelLoadException
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class OCRModel:
    """Manages TrOCR model loading and inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.device = self._get_device()
        self._is_loaded = False
    
    def _get_device(self) -> torch.device:
        """Determine the device to use for model inference"""
        if self.config.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def load_model(self) -> None:
        """Load TrOCR model and processor with fallback support"""
        if self._is_loaded:
            logger.info("TrOCR model already loaded")
            return
        
        try:
            logger.info("Loading TrOCR model...")
            
            # Try Spanish model first
            try:
                self.processor = TrOCRProcessor.from_pretrained(self.config.trocr_spanish_model)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.config.trocr_spanish_model)
                logger.info(f"Spanish TrOCR model loaded successfully: {self.config.trocr_spanish_model}")
            except Exception as e:
                logger.warning(f"Spanish model failed, using fallback: {e}")
                # Fallback to English model
                self.processor = TrOCRProcessor.from_pretrained(self.config.trocr_fallback_model)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.config.trocr_fallback_model)
                logger.info(f"Fallback TrOCR model loaded: {self.config.trocr_fallback_model}")
            
            # Move model to device
            self.model.to(self.device)
            logger.info(f"TrOCR model loaded on {self.device}")
            
            self._is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise ModelLoadException(f"Failed to load TrOCR model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded and self.processor is not None and self.model is not None
    
    def health_check(self) -> dict:
        """Perform health check on the model"""
        return {
            'loaded': self.is_loaded(),
            'device': str(self.device),
            'model_name': self.config.trocr_spanish_model if self.is_loaded() else None,
            'fallback_used': self.is_loaded() and self.processor is not None
        }
    
    def process_image(self, image) -> str:
        """Process image with TrOCR model"""
        if not self.is_loaded():
            self.load_model()
        
        try:
            # Process image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_new_tokens=128)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image with TrOCR: {e}")
            raise ModelLoadException(f"Error processing image: {e}")
    
    def process_batch(self, images) -> list:
        """Process a batch of images"""
        if not self.is_loaded():
            self.load_model()
        
        results = []
        for i, image in enumerate(images):
            try:
                text = self.process_image(image)
                results.append({
                    'index': i,
                    'text': text,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    'index': i,
                    'text': '',
                    'success': False,
                    'error': str(e)
                })
        
        return results
