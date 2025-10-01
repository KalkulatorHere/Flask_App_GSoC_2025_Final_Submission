"""
API routes for Flask OCR Application
"""

import logging
import os
import time
from typing import Dict, Any
from flask import Blueprint, request, send_from_directory, send_file
from werkzeug.utils import secure_filename

from ..exceptions import ValidationException, FileProcessingException, OCRException
from ..config import AppConfig
from ..services import OCRService, LLMService, PDFService, ImageService, FileService
from .handlers import RequestHandler, ResponseHandler

logger = logging.getLogger(__name__)


def create_api_routes(config: AppConfig, services: Dict[str, Any]) -> Blueprint:
    """Create and configure API routes"""
    
    # Initialize services
    ocr_service = services['ocr_service']
    llm_service = services['llm_service']
    pdf_service = services['pdf_service']
    image_service = services['image_service']
    file_service = services['file_service']
    
    # Initialize handlers
    request_handler = RequestHandler(config)
    response_handler = ResponseHandler()
    
    # Create blueprint
    api = Blueprint('api', __name__)
    
    @api.route('/')
    def home():
        """Serve the main HTML file directly"""
        try:
            # Look for the HTML file in the current directory
            html_files = ['index.html', 'paste.html', 'main.html']
            for html_file in html_files:
                if os.path.exists(html_file):
                    return send_file(html_file)
            
            # If no HTML file found, return a basic error page
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>HTML file not found</h1>
                <p>Please ensure your HTML file is in the same directory as app.py</p>
                <p>Expected files: index.html, paste.html, or main.html</p>
            </body>
            </html>
            """, 404
        except Exception as e:
            logger.error(f"Error serving HTML: {e}")
            return response_handler.server_error_response(f"Error serving HTML: {str(e)}")
    
    @api.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and processing"""
        try:
            logger.info("Upload route called")
            
            # Get file and form data
            file, filename = request_handler.get_file_from_request()
            form_data = request_handler.get_form_data()
            
            # Generate unique filename
            unique_filename = file_service.generate_unique_filename(filename)
            file_path = file_service.save_uploaded_file(file, unique_filename)
            
            # Check if it's a PDF
            if filename.lower().endswith('.pdf'):
                logger.info("Processing uploaded PDF...")
                pdf_result = pdf_service.process_pdf_upload(file_path, unique_filename)
                if pdf_result.get('success'):
                    return response_handler.success_response(pdf_result)
                else:
                    return response_handler.file_error_response(
                        pdf_result.get('error', 'PDF processing failed'), 500
                    )
            
            # Handle split image processing
            if form_data['split_image']:
                logger.info("Processing image in two halves...")
                split_result = image_service.process_split_image(file_path, unique_filename)
                
                if split_result['success']:
                    return response_handler.success_response({
                        'filename': unique_filename,
                        'split_processing': True,
                        'left_lines': split_result['left_lines'],
                        'right_lines': split_result['right_lines'],
                        'total_lines': split_result['left_lines'] + split_result['right_lines'],
                        'line_segments': split_result['combined_segments'],
                        'corrected_text': split_result['combined_text'],
                        'pipeline': 'split_processing',
                        'llm_processing': split_result.get('llm_processing', False)
                    })
                else:
                    return response_handler.file_error_response(
                        split_result.get('error', 'Split processing failed'), 500
                    )
            
            # Regular line segmentation and OCR
            logger.info("Starting regular line segmentation and OCR...")
            line_ocr_result = ocr_service.perform_line_segmentation_ocr(file_path)
            
            if line_ocr_result['success']:
                # Also perform regular OCR for backward compatibility
                regular_ocr_text = ocr_service.perform_ocr_inference(file_path)
                
                # Use corrected text if available, otherwise use regular OCR text
                corrected_text = line_ocr_result.get('full_raw_text', regular_ocr_text)
                
                # Save inference data
                inference_data = {
                    'image': unique_filename,
                    'original_text': regular_ocr_text,
                    'corrected_text': corrected_text,
                    'line_segments': line_ocr_result['line_segments'],
                    'total_lines': line_ocr_result['total_lines'],
                    'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                    'llm_processing': line_ocr_result.get('llm_processing', False),
                    'timestamp': time.time()
                }
                
                file_service.save_inference_data(unique_filename, inference_data)
                
                return response_handler.success_response({
                    'filename': unique_filename,
                    'inference': regular_ocr_text,
                    'corrected_text': corrected_text,
                    'line_segments': line_ocr_result['line_segments'],
                    'total_lines': line_ocr_result['total_lines'],
                    'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                    'llm_processing': line_ocr_result.get('llm_processing', False)
                })
            else:
                # Fallback to regular OCR
                regular_ocr_text = ocr_service.perform_ocr_inference(file_path)
                inference_data = {
                    'image': unique_filename,
                    'original_text': regular_ocr_text,
                    'corrected_text': regular_ocr_text,
                    'line_segments': [],
                    'total_lines': 0,
                    'pipeline': 'fallback',
                    'llm_processing': False,
                    'timestamp': time.time()
                }
                
                file_service.save_inference_data(unique_filename, inference_data)
                
                return response_handler.success_response({
                    'filename': unique_filename,
                    'inference': regular_ocr_text,
                    'corrected_text': regular_ocr_text,
                    'line_segments': [],
                    'total_lines': 0,
                    'pipeline': 'fallback',
                    'llm_processing': False
                })
        
        except ValidationException as e:
            logger.warning(f"Validation error: {e}")
            return response_handler.validation_error_response([str(e)])
        except FileProcessingException as e:
            logger.error(f"File processing error: {e}")
            return response_handler.file_error_response(str(e))
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/image/<filename>')
    def get_image(filename):
        """Serve images from upload folder"""
        try:
            return send_from_directory(config.files.upload_folder, filename)
        except Exception as e:
            logger.error(f"Error serving image {filename}: {e}")
            return response_handler.not_found_response(f"Image {filename}")
    
    @api.route('/line_segment/<filename>')
    def get_line_segment(filename):
        """Serve line segment images"""
        try:
            return send_from_directory(config.files.line_segments_folder, filename)
        except Exception as e:
            logger.error(f"Error serving line segment {filename}: {e}")
            return response_handler.not_found_response(f"Line segment {filename}")
    
    @api.route('/get_inference/<filename>')
    def get_inference(filename):
        """Get inference data for a specific image"""
        try:
            inference_data = file_service.load_inference_data(filename)
            if inference_data:
                return response_handler.success_response(inference_data)
            else:
                return response_handler.not_found_response(f"Inference data for {filename}")
        except Exception as e:
            logger.error(f"Error loading inference for {filename}: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/update_inference', methods=['POST'])
    def update_inference():
        """Update corrected text for an image"""
        try:
            data = request_handler.validate_json_request(['image', 'corrected_text'])
            
            success = file_service.update_inference_data(
                data['image'], 
                {'corrected_text': data['corrected_text']}
            )
            
            if success:
                return response_handler.success_response({'updated': True})
            else:
                return response_handler.not_found_response("Inference file")
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error updating inference: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/update_line_ocr', methods=['POST'])
    def update_line_ocr():
        """Update OCR text for a specific line"""
        try:
            data = request_handler.validate_json_request(['image', 'line_index', 'corrected_text'])
            
            # Load existing inference data
            inference_data = file_service.load_inference_data(data['image'])
            if not inference_data:
                return response_handler.not_found_response("Inference file")
            
            # Update the specific line's OCR text
            if 'line_segments' in inference_data:
                for segment in inference_data['line_segments']:
                    if segment['line_index'] == data['line_index']:
                        segment['ocr_text'] = data['corrected_text']
                        segment['ocr_text_corrected'] = data['corrected_text']
                        break
                
                # Recompute combined corrected text
                try:
                    joined = []
                    for seg in sorted(inference_data['line_segments'], key=lambda s: s.get('line_index', 0)):
                        joined.append(seg.get('ocr_text_corrected', seg.get('ocr_text', '')))
                    inference_data['corrected_text'] = "\n".join(joined)
                except Exception:
                    pass
            
            # Save updated data
            file_service.save_inference_data(data['image'], inference_data)
            
            return response_handler.success_response({'updated': True})
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error updating line OCR: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/rerun_inference', methods=['POST'])
    def rerun_inference():
        """Re-run OCR inference on an image"""
        try:
            data = request_handler.validate_json_request(['image'])
            filename = data['image']
            
            image_path = os.path.join(config.files.upload_folder, filename)
            if not os.path.exists(image_path):
                return response_handler.not_found_response("Image file")
            
            # Check if client requests split processing on rerun
            split_image = data.get('split_image', False)
            
            if split_image:
                logger.info(f"Re-running split processing for {filename}...")
                split_result = image_service.process_split_image(image_path, filename)
                if not split_result.get('success'):
                    return response_handler.file_error_response(
                        split_result.get('error', 'Split processing failed')
                    )
                
                # Load the just-saved inference for response consistency
                inference_data = file_service.load_inference_data(filename)
                if inference_data:
                    return response_handler.success_response({
                        'inference': inference_data.get('original_text', ''),
                        'corrected_text': inference_data.get('corrected_text', ''),
                        'line_segments': inference_data.get('line_segments', []),
                        'total_lines': inference_data.get('total_lines', 0),
                        'pipeline': inference_data.get('pipeline', 'split_processing'),
                        'llm_processing': inference_data.get('llm_processing', False),
                        'split_processing': True
                    })
            
            # Perform line segmentation and OCR
            logger.info(f"Re-running line segmentation and OCR for {filename}...")
            line_ocr_result = ocr_service.perform_line_segmentation_ocr(image_path)
            
            # Also perform regular OCR
            regular_ocr_text = ocr_service.perform_ocr_inference(image_path)
            
            # Use corrected text if available
            corrected_text = line_ocr_result.get('full_raw_text', regular_ocr_text)
            
            # Update inference file
            inference_data = {
                'image': filename,
                'original_text': regular_ocr_text,
                'corrected_text': corrected_text,
                'line_segments': line_ocr_result.get('line_segments', []),
                'total_lines': line_ocr_result.get('total_lines', 0),
                'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                'llm_processing': line_ocr_result.get('llm_processing', False),
                'timestamp': time.time(),
                'rerun_count': 1
            }
            
            # Check if file exists and increment rerun count
            existing_data = file_service.load_inference_data(filename)
            if existing_data:
                inference_data['rerun_count'] = existing_data.get('rerun_count', 0) + 1
            
            file_service.save_inference_data(filename, inference_data)
            
            return response_handler.success_response({
                'inference': regular_ocr_text,
                'corrected_text': corrected_text,
                'line_segments': line_ocr_result.get('line_segments', []),
                'total_lines': line_ocr_result.get('total_lines', 0),
                'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                'llm_processing': line_ocr_result.get('llm_processing', False),
                'split_processing': False
            })
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error re-running inference: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/apply_llm_correction', methods=['POST'])
    def apply_llm_correction():
        """Apply LLM correction to existing OCR results"""
        try:
            data = request_handler.validate_json_request(['image'])
            filename = data['image']
            
            # Load existing inference data
            inference_data = file_service.load_inference_data(filename)
            if not inference_data:
                return response_handler.not_found_response("Inference data")
            
            # Get line segments
            line_segments = inference_data.get('line_segments', [])
            if not line_segments:
                return response_handler.error_response("No line segments found", "NO_SEGMENTS", 400)
            
            # Apply LLM correction
            logger.info(f"Applying LLM correction to {len(line_segments)} line segments...")
            corrected_segments = llm_service.process_line_segments_with_gemini(line_segments)
            
            # Create full corrected text
            corrected_text_lines = []
            for segment in corrected_segments:
                corrected_text = segment.get('ocr_text_corrected', segment.get('ocr_text', ''))
                if corrected_text.strip():
                    corrected_text_lines.append(corrected_text)
            
            full_corrected_text = "\n".join(corrected_text_lines)
            
            # Update inference data
            inference_data['corrected_text'] = full_corrected_text
            inference_data['line_segments'] = corrected_segments
            inference_data['llm_processing'] = True
            inference_data['llm_correction_timestamp'] = time.time()
            
            # Save updated data
            file_service.save_inference_data(filename, inference_data)
            
            return response_handler.success_response({
                'corrected_text': full_corrected_text,
                'line_segments': corrected_segments,
                'total_lines': len(corrected_segments),
                'llm_processing': True
            })
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error applying LLM correction: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/apply_llm_to_pdf', methods=['POST'])
    def apply_llm_to_pdf():
        """Apply LLM correction to all pages of an existing PDF"""
        try:
            data = request_handler.validate_json_request(['pdf_filename'])
            pdf_filename = data['pdf_filename']
            
            result = pdf_service.apply_llm_to_pdf(pdf_filename)
            
            if result['success']:
                return response_handler.success_response(result)
            else:
                return response_handler.file_error_response(result.get('error', 'PDF processing failed'))
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error applying LLM to PDF: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/save', methods=['POST'])
    def save_annotations():
        """Save annotation data"""
        try:
            data = request_handler.validate_json_request(['image'])
            
            file_service.save_annotation_data(data['image'], data)
            
            logger.info(f"Annotations saved for {data['image']}: {len(data.get('annotations', []))} boxes")
            return response_handler.success_response({'saved': True})
        
        except ValidationException as e:
            return response_handler.validation_error_response([str(e)])
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/health')
    def health_check():
        """Enhanced health check endpoint"""
        try:
            services_health = {
                'ocr_service': ocr_service.health_check(),
                'llm_service': llm_service.health_check(),
                'pdf_service': pdf_service.health_check(),
                'image_service': image_service.health_check(),
                'file_service': file_service.health_check()
            }
            
            return response_handler.health_check_response(services_health)
        
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return response_handler.server_error_response(str(e))
    
    @api.route('/llm_health')
    def llm_health():
        """Health check for LLM backends"""
        try:
            llm_health = llm_service.health_check()
            return response_handler.success_response(llm_health)
        except Exception as e:
            logger.error(f"LLM health check error: {e}")
            return response_handler.server_error_response(str(e))
    
    return api
