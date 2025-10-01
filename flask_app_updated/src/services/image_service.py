"""
Image processing service for splitting, cropping, and preprocessing
"""

import json
import logging
import os
import time
import cv2
import numpy as np
from typing import Dict, Tuple, List

from ..exceptions import ImageProcessingException
from ..config import AppConfig
from .ocr_service import OCRService
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class ImageService:
    """Service for image processing operations"""
    
    def __init__(self, config: AppConfig, ocr_service: OCRService, llm_service: LLMService):
        self.config = config
        self.ocr_service = ocr_service
        self.llm_service = llm_service
    
    def split_image_into_halves(self, image_path: str, filename: str) -> Dict:
        """Split image into left and right halves and save them"""
        try:
            logger.info(f"🔄 Splitting image: {filename}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image for splitting'}
            
            height, width = image.shape[:2]
            mid_point = width // 2
            
            logger.info(f"📏 Image dimensions: {width}x{height}, splitting at {mid_point}")
            
            # Split image into left and right halves
            left_half = image[:, :mid_point]
            right_half = image[:, mid_point:]
            
            # Save split images with unique names
            name, ext = os.path.splitext(filename)
            timestamp = str(int(time.time()))
            left_filename = f"{name}_left_{timestamp}{ext}"
            right_filename = f"{name}_right_{timestamp}{ext}"
            
            left_path = os.path.join(self.config.files.upload_folder, left_filename)
            right_path = os.path.join(self.config.files.upload_folder, right_filename)
            
            # Save the split images
            cv2.imwrite(left_path, left_half)
            cv2.imwrite(right_path, right_half)
            
            logger.info(f"✅ Split images saved:")
            logger.info(f"   Left: {left_path}")
            logger.info(f"   Right: {right_path}")
            
            return {
                'success': True,
                'left_path': left_path,
                'right_path': right_path,
                'left_filename': left_filename,
                'right_filename': right_filename,
                'original_dimensions': (width, height),
                'split_point': mid_point
            }
            
        except Exception as e:
            logger.error(f"❌ Error splitting image: {str(e)}")
            raise ImageProcessingException(f"Image splitting failed: {e}")
    
    def process_split_image(self, image_path: str, filename: str) -> Dict:
        """Process image by splitting it into left and right halves"""
        try:
            logger.info(f"🚀 Starting split processing for {filename}")
            
            # Step 1: Split the image
            split_result = self.split_image_into_halves(image_path, filename)
            if not split_result['success']:
                return {'success': False, 'error': f'Split failed: {split_result["error"]}'}
            
            left_path = split_result['left_path']
            right_path = split_result['right_path']
            left_filename = split_result['left_filename']
            right_filename = split_result['right_filename']
            
            # Step 2: Process left half first
            logger.info("📄 Processing LEFT half...")
            left_result = self.ocr_service.perform_line_segmentation_ocr(left_path)
            
            if not left_result['success']:
                logger.error(f"❌ Left half processing failed: {left_result.get('error', 'Unknown error')}")
                return {'success': False, 'error': f'Left half failed: {left_result.get("error", "Unknown error")}'}
            
            left_segments = left_result.get('line_segments', [])
            logger.info(f"✅ Left half processed: {len(left_segments)} lines")
            
            # Prefix and rename left half line segment images to avoid filename collisions
            try:
                for seg in left_segments:
                    img_name = seg.get('image_filename')
                    if not img_name:
                        continue
                    # Avoid double prefixing
                    if not img_name.startswith('left_'):
                        old_path = os.path.join(self.config.files.line_segments_folder, img_name)
                        new_name = f"left_{img_name}"
                        new_path = os.path.join(self.config.files.line_segments_folder, new_name)
                        if os.path.exists(old_path):
                            try:
                                os.replace(old_path, new_path)
                            except Exception:
                                pass
                        seg['image_filename'] = new_name
            except Exception as e:
                logger.warning(f"Warning: could not rename left half line segment images: {e}")
            
            # Step 3: Process right half
            logger.info("📄 Processing RIGHT half...")
            right_result = self.ocr_service.perform_line_segmentation_ocr(right_path)
            
            if not right_result['success']:
                logger.error(f"❌ Right half processing failed: {right_result.get('error', 'Unknown error')}")
                return {'success': False, 'error': f'Right half failed: {right_result.get("error", "Unknown error")}'}
            
            right_segments = right_result.get('line_segments', [])
            logger.info(f"✅ Right half processed: {len(right_segments)} lines")
            
            # Prefix and rename right half line segment images to avoid filename collisions
            try:
                for seg in right_segments:
                    img_name = seg.get('image_filename')
                    if not img_name:
                        continue
                    # Avoid double prefixing
                    if not img_name.startswith('right_'):
                        old_path = os.path.join(self.config.files.line_segments_folder, img_name)
                        new_name = f"right_{img_name}"
                        new_path = os.path.join(self.config.files.line_segments_folder, new_name)
                        if os.path.exists(old_path):
                            try:
                                os.replace(old_path, new_path)
                            except Exception:
                                pass
                        seg['image_filename'] = new_name
            except Exception as e:
                logger.warning(f"Warning: could not rename right half line segment images: {e}")
            
            # Step 4: Combine results
            logger.info(f"🔗 Combining {len(left_segments)} left + {len(right_segments)} right segments")
            
            # Adjust line indices for right half to continue from left half
            for segment in right_segments:
                segment['line_index'] += len(left_segments)
                segment['reading_order_index'] += len(left_segments)
                segment['position_in_column'] += len(left_segments)
                # image_filename already prefixed and renamed earlier
            
            # Combine segments
            combined_segments = left_segments + right_segments
            
            # Combine text from results
            left_text = left_result.get('full_raw_text', '')
            right_text = right_result.get('full_raw_text', '')
            combined_text = left_text + '\n' + right_text if left_text and right_text else left_text + right_text
            
            # Fallback: build from segments if combined_text is empty
            if not combined_text:
                try:
                    combined_text_lines = []
                    for seg in combined_segments:
                        line_txt = seg.get('ocr_text_corrected', seg.get('ocr_text', ''))
                        if line_txt.strip():
                            combined_text_lines.append(line_txt)
                    combined_text = "\n".join(combined_text_lines)
                except Exception as e:
                    logger.warning(f"⚠️ Error building combined text from segments: {e}")
                    combined_text = ''
            
            logger.info(f"📝 Combined text length: {len(combined_text)} characters")
            
            # Step 5: Save inference data
            inference_data = {
                'image': filename,
                'split_processing': True,
                'left_half': left_filename,
                'right_half': right_filename,
                'left_lines': len(left_segments),
                'right_lines': len(right_segments),
                'total_lines': len(combined_segments),
                'line_segments': combined_segments,
                'combined_text': combined_text,
                'left_text': left_text,
                'right_text': right_text,
                'original_text': combined_text,
                'corrected_text': combined_text,
                'pipeline': 'split_processing',
                'llm_processing': left_result.get('llm_processing', False) or right_result.get('llm_processing', False),
                'timestamp': time.time(),
                'split_info': {
                    'original_dimensions': split_result['original_dimensions'],
                    'split_point': split_result['split_point'],
                    'left_filename': left_filename,
                    'right_filename': right_filename
                }
            }
            
            inference_path = os.path.join(self.config.files.inference_folder, f"{filename}.json")
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)

            # Also save separate inference files for each half to enable per-page viewing
            try:
                left_inference = {
                    'image': left_filename,
                    'parent_image': filename,
                    'split_side': 'left',
                    'split_processing': True,
                    'original_text': left_text,
                    'corrected_text': left_text,
                    'line_segments': left_segments,
                    'total_lines': len(left_segments),
                    'pipeline': left_result.get('pipeline', 'split_processing'),
                    'llm_processing': left_result.get('llm_processing', False),
                    'timestamp': time.time()
                }
                with open(os.path.join(self.config.files.inference_folder, f"{left_filename}.json"), 'w', encoding='utf-8') as f:
                    json.dump(left_inference, f, ensure_ascii=False, indent=2)

                right_inference = {
                    'image': right_filename,
                    'parent_image': filename,
                    'split_side': 'right',
                    'split_processing': True,
                    'original_text': right_text,
                    'corrected_text': right_text,
                    'line_segments': right_segments,
                    'total_lines': len(right_segments),
                    'pipeline': right_result.get('pipeline', 'split_processing'),
                    'llm_processing': right_result.get('llm_processing', False),
                    'timestamp': time.time()
                }
                with open(os.path.join(self.config.files.inference_folder, f"{right_filename}.json"), 'w', encoding='utf-8') as f:
                    json.dump(right_inference, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Warning: could not save separate half inference files: {e}")
            
            logger.info(f"💾 Split processing completed successfully!")
            logger.info(f"   Saved to: {inference_path}")
            logger.info(f"   Total lines: {len(combined_segments)}")
            logger.info(f"   LLM processing: {inference_data['llm_processing']}")
            
            return {
                'success': True,
                'left_lines': len(left_segments),
                'right_lines': len(right_segments),
                'combined_segments': combined_segments,
                'combined_text': combined_text,
                'left_segments': left_segments,
                'right_segments': right_segments,
                'left_text': left_text,
                'right_text': right_text,
                'left_image': left_filename,
                'right_image': right_filename,
                'llm_processing': inference_data['llm_processing'],
                'split_info': split_result
            }
            
        except Exception as e:
            logger.error(f"❌ Split image processing error: {str(e)}")
            raise ImageProcessingException(f"Split image processing failed: {e}")
    
    def validate_image(self, image_path: str) -> bool:
        """Validate that the image can be loaded and processed"""
        try:
            image = cv2.imread(image_path)
            return image is not None
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ImageProcessingException("Could not load image")
            height, width = image.shape[:2]
            return width, height
        except Exception as e:
            raise ImageProcessingException(f"Could not get image dimensions: {e}")
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check on image service"""
        return {
            'service_ready': True,
            'upload_folder_exists': os.path.exists(self.config.files.upload_folder),
            'line_segments_folder_exists': os.path.exists(self.config.files.line_segments_folder)
        }
