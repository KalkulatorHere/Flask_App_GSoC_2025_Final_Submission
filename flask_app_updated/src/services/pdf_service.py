"""
PDF processing service for multi-page document handling
"""

import logging
import os
import time
import json
from typing import List, Dict, Optional
import cv2
import numpy as np
import fitz  # PyMuPDF

from ..exceptions import PDFProcessingException
from ..config import AppConfig
from .ocr_service import OCRService
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class PDFService:
    """Service for PDF processing and multi-page document handling"""
    
    def __init__(self, config: AppConfig, ocr_service: OCRService, llm_service: LLMService):
        self.config = config
        self.ocr_service = ocr_service
        self.llm_service = llm_service
    
    def convert_pdf_to_images(self, pdf_path: str, output_base_name: str, dpi: int = 200) -> List[Dict]:
        """Render a PDF into page images saved in UPLOAD_FOLDER.

        Returns a list of dicts with keys: page_index, image_filename, image_path
        """
        page_infos: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for i in range(total_pages):
                page = doc.load_page(i)
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                page_filename = f"{output_base_name}_page_{i+1:03d}.png"
                page_path = os.path.join(self.config.files.upload_folder, page_filename)
                cv2.imwrite(page_path, img)
                page_infos.append({
                    'page_index': i,
                    'image_filename': page_filename,
                    'image_path': page_path
                })
            doc.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise PDFProcessingException(f"PDF conversion failed: {e}")
        return page_infos
    
    def process_pdf_upload(self, pdf_path: str, pdf_filename: str) -> Dict:
        """Process a PDF by rendering each page to an image and running the pipeline per page."""
        try:
            base_name, _ = os.path.splitext(pdf_filename)
            page_infos = self.convert_pdf_to_images(pdf_path, base_name, dpi=200)
            total_pages = len(page_infos)
            page_filenames: List[str] = []

            for info in page_infos:
                page_filename = info['image_filename']
                page_path = info['image_path']
                page_index = info['page_index']

                # Run line segmentation OCR pipeline per page image
                line_ocr_result = self.ocr_service.perform_line_segmentation_ocr(page_path)
                
                # Ensure line segment crop filenames are unique per page
                try:
                    page_stem = os.path.splitext(page_filename)[0]
                    if line_ocr_result.get('line_segments'):
                        for seg in line_ocr_result['line_segments']:
                            old_name = seg.get('image_filename')
                            if not old_name:
                                continue
                            old_path = os.path.join(self.config.files.line_segments_folder, old_name)
                            new_name = f"{page_stem}__{old_name}"
                            new_path = os.path.join(self.config.files.line_segments_folder, new_name)
                            if os.path.exists(old_path):
                                try:
                                    os.replace(old_path, new_path)
                                except Exception:
                                    pass
                            seg['image_filename'] = new_name
                except Exception as e:
                    logger.warning(f"Warning renaming line segment images for {page_filename}: {e}")
                
                # Also run regular OCR for backward compatibility and full text
                regular_ocr_text = self.ocr_service.perform_ocr_inference(page_path)
                corrected_text = line_ocr_result.get('full_raw_text', regular_ocr_text)

                # Save per-page inference data
                inference_data = {
                    'image': page_filename,
                    'pdf_parent': pdf_filename,
                    'page_index': page_index,
                    'total_pages': total_pages,
                    'original_text': regular_ocr_text,
                    'corrected_text': corrected_text,
                    'line_segments': line_ocr_result.get('line_segments', []),
                    'total_lines': line_ocr_result.get('total_lines', 0),
                    'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                    'llm_processing': line_ocr_result.get('llm_processing', False),
                    'timestamp': time.time(),
                    'is_pdf_page': True
                }

                inference_path = os.path.join(self.config.files.inference_folder, f"{page_filename}.json")
                with open(inference_path, 'w', encoding='utf-8') as f:
                    json.dump(inference_data, f, ensure_ascii=False, indent=2)

                page_filenames.append(page_filename)

            # Optionally save a manifest for the PDF
            try:
                manifest = {
                    'pdf_filename': pdf_filename,
                    'total_pages': total_pages,
                    'page_filenames': page_filenames,
                    'created_at': time.time()
                }
                manifest_path = os.path.join(self.config.files.inference_folder, f"{base_name}_manifest.json")
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Warning: could not save PDF manifest: {e}")

            return {
                'success': True,
                'is_pdf': True,
                'filename': pdf_filename,
                'page_filenames': page_filenames,
                'first_page': page_filenames[0] if page_filenames else None,
                'total_pages': total_pages
            }
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise PDFProcessingException(f"PDF processing failed: {e}")
    
    def apply_llm_to_pdf(self, pdf_filename: str) -> Dict:
        """Apply LLM correction to all pages of an existing PDF"""
        try:
            # Load PDF manifest to get page filenames
            base_name, _ = os.path.splitext(pdf_filename)
            manifest_path = os.path.join(self.config.files.inference_folder, f"{base_name}_manifest.json")
            
            if not os.path.exists(manifest_path):
                raise PDFProcessingException("PDF manifest not found")
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            page_filenames = manifest.get('page_filenames', [])
            if not page_filenames:
                raise PDFProcessingException("No pages found in manifest")
            
            # Apply LLM correction to all pages
            logger.info(f"Applying LLM correction to {len(page_filenames)} pages of {pdf_filename}...")
            previous_context = ""
            processed_pages = 0
            
            for i, page_filename in enumerate(page_filenames):
                logger.info(f"Processing page {i + 1}/{len(page_filenames)}: {page_filename}")
                
                inference_path = os.path.join(self.config.files.inference_folder, f"{page_filename}.json")
                
                if not os.path.exists(inference_path):
                    logger.warning(f"Warning: Inference file not found for {page_filename}")
                    continue
                
                # Load existing inference data
                with open(inference_path, 'r', encoding='utf-8') as f:
                    inference_data = json.load(f)
                
                original_text = inference_data.get('original_text', '')
                
                if original_text.strip():
                    # Apply LLM correction with context from previous page
                    corrected_text, status = self.llm_service.process_text_with_fallbacks(original_text, previous_context)
                    
                    # Update inference data with LLM results
                    inference_data['corrected_text'] = corrected_text
                    inference_data['llm_processing'] = (status in ["gemini", "hf", "local_llama"])
                    inference_data['llm_correction_status'] = status
                    inference_data['llm_correction_timestamp'] = time.time()
                    inference_data['llm_reprocessed'] = True
                    
                    # Update context for next page
                    if status in ["gemini", "hf", "local_llama"]:
                        previous_context = self.llm_service.get_last_two_lines(corrected_text)
                    else:
                        previous_context = self.llm_service.get_last_two_lines(original_text)
                    
                    processed_pages += 1
                else:
                    # Empty text - mark as processed
                    inference_data['llm_processing'] = True
                    inference_data['llm_correction_status'] = "empty_text"
                    inference_data['llm_correction_timestamp'] = time.time()
                    inference_data['llm_reprocessed'] = True
                
                # Save updated inference data
                with open(inference_path, 'w', encoding='utf-8') as f:
                    json.dump(inference_data, f, ensure_ascii=False, indent=2)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            # Update manifest
            manifest['llm_processing_completed'] = True
            manifest['llm_reprocessed_at'] = time.time()
            manifest['processed_pages_count'] = processed_pages
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            
            return {
                'success': True,
                'pdf_filename': pdf_filename,
                'total_pages': len(page_filenames),
                'processed_pages': processed_pages,
                'llm_processing_completed': True
            }
        
        except Exception as e:
            logger.error(f"Error applying LLM to PDF: {e}")
            raise PDFProcessingException(f"LLM processing failed: {e}")
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check on PDF service"""
        return {
            'service_ready': True,
            'upload_folder_exists': os.path.exists(self.config.files.upload_folder),
            'inference_folder_exists': os.path.exists(self.config.files.inference_folder),
            'line_segments_folder_exists': os.path.exists(self.config.files.line_segments_folder)
        }
