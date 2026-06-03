"""
OCR Processing Service - Handles the main OCR pipeline and processing logic
"""
import os
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .ocr_models import OCRModelsService
from .llm_service import LLMService
from .utils import (
    split_image_into_halves, convert_pdf_to_images, create_inference_data,
    update_line_segment_filenames, adjust_line_indices_for_continuation,
    combine_text_from_segments, save_json_data
)

logger = logging.getLogger(__name__)

class OCRProcessingService:
    """Service for handling OCR processing pipeline"""
    
    def __init__(self, config, ocr_models_service: OCRModelsService, llm_service: LLMService):
        self.config = config
        self.ocr_models = ocr_models_service
        self.llm_service = llm_service
        self.current_image_index = 0
    
    def perform_line_segmentation_ocr(self, image_path: str, skip_gemini: bool = False) -> Dict:
        """Enhanced line segmentation and OCR using advanced pipeline with optional Gemini correction"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Try advanced pipeline first
            if self.ocr_models.textline_extractor is not None:
                try:
                    print("Using advanced textline extraction pipeline...")
                    
                    # Extract textlines
                    boxes, scores, outputs = self.ocr_models.textline_extractor.extract_textlines(image)
                    
                    if len(boxes) == 0:
                        return {'success': False, 'error': 'No textlines detected'}
                    
                    # Filter by area and sort in reading order
                    filtered_boxes, filtered_scores, margin_boxes, margin_scores = self.ocr_models.textline_extractor.filter_margin_boxes_by_area(boxes, scores)
                    
                    if len(filtered_boxes) == 0:
                        return {'success': False, 'error': 'No textlines after filtering'}
                    
                    # Sort in reading order
                    ordered_boxes, ordered_scores, reading_order_info = self.ocr_models.textline_extractor.detect_columns_and_sort_reading_order(
                        filtered_boxes, filtered_scores
                    )
                    
                    # Crop textlines with dynamic padding
                    cropped_textlines, padded_boxes, padding_info = self.ocr_models.textline_extractor.crop_textlines_with_dynamic_padding(
                        image, ordered_boxes, use_margin_filtering=False
                    )
                    
                    # Process with TrOCR in sequential order
                    ocr_results = self.ocr_models.textline_extractor.process_textlines_with_trocr(cropped_textlines, reading_order_info)
                    
                    # Save line segment images and prepare results
                    line_segments = []
                    for i, (crop, ocr_result, bbox, padded_bbox, score) in enumerate(zip(
                        cropped_textlines, ocr_results, ordered_boxes, padded_boxes, ordered_scores
                    )):
                        # Save crop image
                        crop_filename = f"line_{i:03d}_reading_order_{ocr_result['reading_order_index']:03d}.png"
                        crop_path = os.path.join(self.config.line_segments_folder, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        
                        # Convert NumPy values to native Python types
                        bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else [float(x) for x in bbox]
                        padded_bbox_list = [float(x) for x in padded_bbox]
                        
                        line_segments.append({
                            'line_index': int(i),
                            'reading_order_index': int(ocr_result['reading_order_index']),
                            'column': int(ocr_result['column']),
                            'position_in_column': int(ocr_result['position_in_column']),
                            'image_filename': crop_filename,
                            'bbox': bbox_list,
                            'padded_bbox': padded_bbox_list,
                            'score': float(score),
                            'ocr_text': str(ocr_result['text']),
                            'ocr_text_pre_llm': str(ocr_result['text']),  # snapshot before LLM
                            'ocr_text_corrected': '',
                            'ocr_text_manual': '',
                            'confidence': float(ocr_result['confidence']),
                            'manually_edited': False
                        })
                    
                    # Sort line segments by reading order for final output
                    line_segments.sort(key=lambda x: x['reading_order_index'])
                    
                    # Apply LLM post-processing if available and not skipped
                    # Assemble pre-LLM text before correction runs
                    pre_llm_text_lines = [
                        s.get('ocr_text', '') for s in line_segments
                        if s.get('ocr_text', '').strip()
                    ]
                    full_pre_llm_text = "\n".join(pre_llm_text_lines)

                    if hasattr(self.llm_service, 'process_line_segments_with_gemini') and not skip_gemini:
                        print("Applying LLM text correction...")
                        corrected_segments = self.llm_service.process_line_segments_with_gemini(line_segments)

                        corrected_text_lines = [
                            s.get('ocr_text_corrected', s.get('ocr_text', ''))
                            for s in corrected_segments
                            if s.get('ocr_text_corrected', s.get('ocr_text', '')).strip()
                        ]
                        full_corrected_text = "\n".join(corrected_text_lines)

                        return {
                            'success': True,
                            'line_segments': corrected_segments,
                            'total_lines': len(corrected_segments),
                            'pipeline': 'advanced_with_llm',
                            'padding_info': padding_info,
                            'pre_llm_text': full_pre_llm_text,
                            'full_corrected_text': full_corrected_text,
                            'gemini_processing': True,
                            'llm_corrected': True
                        }
                    else:
                        return {
                            'success': True,
                            'line_segments': line_segments,
                            'total_lines': len(line_segments),
                            'pipeline': 'advanced_no_llm' if skip_gemini else 'advanced',
                            'padding_info': padding_info,
                            'pre_llm_text': full_pre_llm_text,
                            'full_raw_text': full_pre_llm_text,
                            'gemini_processing': False,
                            'llm_corrected': False
                        }
                    
                except Exception as e:
                    print(f"Advanced pipeline failed: {e}")
                    # Fall back to original method
            
            # Fallback to original method
            print("Using fallback textline extraction...")
            boxes, scores, image = self.ocr_models.extract_textlines_from_image(image_path)
            
            if len(boxes) == 0:
                return {'success': False, 'error': 'No textlines detected'}
            
            # Use original cropping method
            cropped_textlines, padded_boxes = self.ocr_models.crop_textlines_with_padding(image, boxes)
            
            # Use original OCR method
            ocr_results = self.ocr_models.process_textlines_with_trocr(cropped_textlines)
            
            # Save line segment images
            line_segments = []
            for i, (crop, ocr_result) in enumerate(zip(cropped_textlines, ocr_results)):
                crop_filename = f"line_{i:03d}.png"
                crop_path = os.path.join(self.config.line_segments_folder, crop_filename)
                cv2.imwrite(crop_path, crop)
                
                # Convert NumPy values to native Python types
                bbox_list = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else [float(x) for x in boxes[i]]
                padded_bbox_list = [float(x) for x in padded_boxes[i]]
                
                line_segments.append({
                    'line_index': int(i),
                    'reading_order_index': int(i),
                    'column': 0,
                    'position_in_column': int(i),
                    'image_filename': crop_filename,
                    'bbox': bbox_list,
                    'padded_bbox': padded_bbox_list,
                    'score': float(scores[i]),
                    'ocr_text': str(ocr_result['text']),
                    'ocr_text_pre_llm': str(ocr_result['text']),  # snapshot before LLM
                    'ocr_text_corrected': '',
                    'ocr_text_manual': '',
                    'confidence': float(ocr_result['confidence']),
                    'manually_edited': False
                })
            
            # Apply LLM post-processing if available and not skipped
            # Assemble pre-LLM text before correction runs
            pre_llm_text_lines = [
                s.get('ocr_text', '') for s in line_segments
                if s.get('ocr_text', '').strip()
            ]
            full_pre_llm_text = "\n".join(pre_llm_text_lines)

            if hasattr(self.llm_service, 'process_line_segments_with_gemini') and not skip_gemini:
                print("Applying LLM text correction to fallback results...")
                corrected_segments = self.llm_service.process_line_segments_with_gemini(line_segments)

                corrected_text_lines = [
                    s.get('ocr_text_corrected', s.get('ocr_text', ''))
                    for s in corrected_segments
                    if s.get('ocr_text_corrected', s.get('ocr_text', '')).strip()
                ]
                full_corrected_text = "\n".join(corrected_text_lines)

                return {
                    'success': True,
                    'line_segments': corrected_segments,
                    'total_lines': len(corrected_segments),
                    'pipeline': 'fallback_with_llm',
                    'pre_llm_text': full_pre_llm_text,
                    'full_corrected_text': full_corrected_text,
                    'gemini_processing': True,
                    'llm_corrected': True
                }
            else:
                return {
                    'success': True,
                    'line_segments': line_segments,
                    'total_lines': len(line_segments),
                    'pipeline': 'fallback_no_llm' if skip_gemini else 'fallback',
                    'pre_llm_text': full_pre_llm_text,
                    'full_raw_text': full_pre_llm_text,
                    'gemini_processing': False,
                    'llm_corrected': False
                }
                        
        except Exception as e:
            print(f"Line segmentation OCR error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def perform_line_segmentation_ocr_no_gemini(self, image_path: str) -> Dict:
        """Enhanced line segmentation and OCR using advanced pipeline WITHOUT LLM correction"""
        return self.perform_line_segmentation_ocr(image_path, skip_gemini=True)
    
    def process_split_image(self, image_path: str, filename: str) -> Dict:
        """Process image by splitting it into left and right halves"""
        try:
            print(f"🚀 Starting split processing for {filename}")
            
            # Step 1: Split the image
            split_result = split_image_into_halves(image_path, filename)
            if not split_result['success']:
                return {'success': False, 'error': f'Split failed: {split_result["error"]}'}
            
            left_path = split_result['left_path']
            right_path = split_result['right_path']
            left_filename = split_result['left_filename']
            right_filename = split_result['right_filename']
            
            # Step 2: Process left half first
            print("📄 Processing LEFT half...")
            left_result = self.perform_line_segmentation_ocr(left_path)
            
            if not left_result['success']:
                print(f"❌ Left half processing failed: {left_result.get('error', 'Unknown error')}")
                return {'success': False, 'error': f'Left half failed: {left_result.get("error", "Unknown error")}'}
            
            left_segments = left_result.get('line_segments', [])
            print(f"✅ Left half processed: {len(left_segments)} lines")
            
            # Prefix and rename left half line segment images to avoid filename collisions
            update_line_segment_filenames(left_segments, 'left', self.config.line_segments_folder)
            
            # Step 3: Process right half
            print("📄 Processing RIGHT half...")
            right_result = self.perform_line_segmentation_ocr(right_path)
            
            if not right_result['success']:
                print(f"❌ Right half processing failed: {right_result.get('error', 'Unknown error')}")
                return {'success': False, 'error': f'Right half failed: {right_result.get("error", "Unknown error")}'}
            
            right_segments = right_result.get('line_segments', [])
            print(f"✅ Right half processed: {len(right_segments)} lines")
            
            # Prefix and rename right half line segment images to avoid filename collisions
            update_line_segment_filenames(right_segments, 'right', self.config.line_segments_folder)
            
            # Step 4: Combine results
            print(f"🔗 Combining {len(left_segments)} left + {len(right_segments)} right segments")
            
            # Adjust line indices for right half to continue from left half
            adjust_line_indices_for_continuation(right_segments, len(left_segments))
            
            # Combine segments
            combined_segments = left_segments + right_segments
            
            # Combine text from results
            left_text = left_result.get('full_corrected_text', combine_text_from_segments(left_segments))
            right_text = right_result.get('full_corrected_text', combine_text_from_segments(right_segments))
            combined_text = left_text + '\n' + right_text if left_text and right_text else left_text + right_text
            
            # Fallback: build from segments if combined_text is empty
            if not combined_text:
                combined_text = combine_text_from_segments(combined_segments)
            
            print(f"📝 Combined text length: {len(combined_text)} characters")
            
            # Step 5: Save inference data
            inference_data = create_inference_data(
                filename=filename,
                original_text=combined_text,
                corrected_text=combined_text,
                line_segments=combined_segments,
                total_lines=len(combined_segments),
                pipeline='split_processing',
                gemini_processing=left_result.get('gemini_processing', False) or right_result.get('gemini_processing', False),
                split_processing=True,
                left_half=left_filename,
                right_half=right_filename,
                left_lines=len(left_segments),
                right_lines=len(right_segments),
                split_info={
                    'original_dimensions': split_result['original_dimensions'],
                    'split_point': split_result['split_point'],
                    'left_filename': left_filename,
                    'right_filename': right_filename
                }
            )
            
            inference_path = os.path.join(self.config.inference_folder, f"{filename}.json")
            save_json_data(inference_data, inference_path)
            
            # Also save separate inference files for each half to enable per-page viewing
            try:
                left_inference = create_inference_data(
                    filename=left_filename,
                    original_text=left_text,
                    corrected_text=left_text,
                    line_segments=left_segments,
                    total_lines=len(left_segments),
                    pipeline=left_result.get('pipeline', 'split_processing'),
                    gemini_processing=left_result.get('gemini_processing', False),
                    parent_image=filename,
                    split_side='left',
                    split_processing=True
                )
                save_json_data(left_inference, os.path.join(self.config.inference_folder, f"{left_filename}.json"))
                
                right_inference = create_inference_data(
                    filename=right_filename,
                    original_text=right_text,
                    corrected_text=right_text,
                    line_segments=right_segments,
                    total_lines=len(right_segments),
                    pipeline=right_result.get('pipeline', 'split_processing'),
                    gemini_processing=right_result.get('gemini_processing', False),
                    parent_image=filename,
                    split_side='right',
                    split_processing=True
                )
                save_json_data(right_inference, os.path.join(self.config.inference_folder, f"{right_filename}.json"))
            except Exception as e:
                print(f"Warning: could not save separate half inference files: {e}")
            
            print(f"💾 Split processing completed successfully!")
            print(f"   Saved to: {inference_path}")
            print(f"   Total lines: {len(combined_segments)}")
            print(f"   LLM processing: {inference_data['gemini_processing']}")
            
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
                'gemini_processing': inference_data['gemini_processing'],
                'split_info': split_result
            }
            
        except Exception as e:
            print(f"❌ Split image processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def process_pdf_upload(self, pdf_path: str, pdf_filename: str) -> Dict:
        """Process a PDF by rendering each page to an image and running the pipeline per page."""
        try:
            base_name, _ = os.path.splitext(pdf_filename)
            page_infos = convert_pdf_to_images(pdf_path, base_name, self.config.upload_folder, dpi=200)
            total_pages = len(page_infos)
            page_filenames: List[str] = []

            for info in page_infos:
                page_filename = info['image_filename']
                page_path = info['image_path']
                page_index = info['page_index']

                # Run line segmentation OCR pipeline per page image
                line_ocr_result = self.perform_line_segmentation_ocr(page_path)
                
                # Ensure line segment crop filenames are unique per page
                try:
                    page_stem = os.path.splitext(page_filename)[0]
                    if line_ocr_result.get('line_segments'):
                        for seg in line_ocr_result['line_segments']:
                            old_name = seg.get('image_filename')
                            if not old_name:
                                continue
                            old_path = os.path.join(self.config.line_segments_folder, old_name)
                            new_name = f"{page_stem}__{old_name}"
                            new_path = os.path.join(self.config.line_segments_folder, new_name)
                            if os.path.exists(old_path):
                                try:
                                    os.replace(old_path, new_path)
                                except Exception:
                                    pass
                            seg['image_filename'] = new_name
                except Exception as e:
                    print(f"Warning renaming line segment images for {page_filename}: {e}")
                
                # Also run regular OCR for backward compatibility and full text
                regular_ocr_text = self.ocr_models.perform_ocr_inference(page_path)
                corrected_text = line_ocr_result.get('full_corrected_text', regular_ocr_text)

                # Save per-page inference data
                inference_data = create_inference_data(
                    filename=page_filename,
                    original_text=regular_ocr_text,
                    pre_llm_text=line_ocr_result.get('pre_llm_text', regular_ocr_text),
                    corrected_text=corrected_text,
                    manual_text='',
                    line_segments=line_ocr_result.get('line_segments', []),
                    total_lines=line_ocr_result.get('total_lines', 0),
                    pipeline=line_ocr_result.get('pipeline', 'unknown'),
                    gemini_processing=line_ocr_result.get('gemini_processing', False),
                    llm_corrected=line_ocr_result.get('llm_corrected', False),
                    manually_edited=False,
                    pdf_parent=pdf_filename,
                    page_index=page_index,
                    total_pages=total_pages,
                    is_pdf_page=True
                )

                inference_path = os.path.join(self.config.inference_folder, f"{page_filename}.json")
                save_json_data(inference_data, inference_path)

                page_filenames.append(page_filename)

            # Optionally save a manifest for the PDF
            try:
                manifest = {
                    'pdf_filename': pdf_filename,
                    'total_pages': total_pages,
                    'page_filenames': page_filenames,
                    'created_at': time.time()
                }
                manifest_path = os.path.join(self.config.inference_folder, f"{base_name}_manifest.json")
                save_json_data(manifest, manifest_path)
            except Exception as e:
                print(f"Warning: could not save PDF manifest: {e}")

            return {
                'success': True,
                'is_pdf': True,
                'filename': pdf_filename,
                'page_filenames': page_filenames,
                'first_page': page_filenames[0] if page_filenames else None,
                'total_pages': total_pages
            }
        except Exception as e:
            print(f"PDF processing error: {e}")
            return {'success': False, 'error': str(e)}
