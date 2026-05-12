"""
Flask Routes - All API endpoints for the OCR application
"""
import os
import time
import logging
from flask import Blueprint, request, jsonify, send_from_directory, send_file

from .utils import (
    allowed_file, is_image_file, generate_unique_filename, save_json_data, 
    load_json_data, create_inference_data, combine_text_from_segments,
    get_last_two_lines, sanitize_filename
)

logger = logging.getLogger(__name__)

def create_routes(config, ocr_processing_service, llm_service) -> Blueprint:
    """Create Flask routes blueprint"""
    routes_bp = Blueprint('routes', __name__)
    
    @routes_bp.route('/')
    @routes_bp.route('/')
    def home():
        """Serve the main HTML file directly"""
        try:
            # Get the directory where routes.py lives
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            html_files = ['index.html', 'paste.html', 'main.html']
            for html_file in html_files:
                full_path = os.path.join(base_dir, html_file)
                if os.path.exists(full_path):
                    return send_file(full_path)
            
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
            return f"Error serving HTML: {str(e)}", 500

    @routes_bp.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and processing"""
        print("Upload route called")
        print("Files in request:", request.files)
        print("Form data:", request.form)
        
        # Check if file part exists
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400

        file = request.files['file']
        split_image = request.form.get('split_image', 'false').lower() == 'true'

        # Validate filename
        if file.filename == '':
            print("No selected file")
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Validate extension
        if not allowed_file(file.filename, config.allowed_extensions):
            print("Invalid file type:", file.filename)
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        try:
            # Save file
            filename = sanitize_filename(file.filename)
            filename = generate_unique_filename(filename)
            
            file_path = os.path.join(config.upload_folder, filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")

            # Special handling for PDF uploads: render each page and process per-page
            if filename.lower().endswith('.pdf'):
                print("Processing uploaded PDF...")
                pdf_result = ocr_processing_service.process_pdf_upload(file_path, filename)
                if pdf_result.get('success'):
                    return jsonify(pdf_result)
                else:
                    return jsonify({'success': False, 'error': pdf_result.get('error', 'PDF processing failed')}), 500

            if split_image:
                # Process image in two halves
                print("Processing image in two halves...")
                split_result = ocr_processing_service.process_split_image(file_path, filename)
                
                if split_result['success']:
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'split_processing': True,
                        'left_lines': split_result['left_lines'],
                        'right_lines': split_result['right_lines'],
                        'total_lines': split_result['left_lines'] + split_result['right_lines'],
                        'line_segments': split_result['combined_segments'],
                        'corrected_text': split_result['combined_text'],
                        'pipeline': 'split_processing',
                        'gemini_processing': split_result.get('gemini_processing', False)
                    })
                else:
                    # Fallback to regular processing
                    print("Split processing failed, falling back to regular processing")
                    split_image = False

            if not split_image:
                # Perform regular line segmentation and OCR
                print("Starting regular line segmentation and OCR...")
                line_ocr_result = ocr_processing_service.perform_line_segmentation_ocr(file_path)
                
                if line_ocr_result['success']:
                    # Also perform regular OCR for backward compatibility
                    regular_ocr_text = ocr_processing_service.ocr_models.perform_ocr_inference(file_path)
                    
                    # Use LLM-corrected text if available, otherwise use regular OCR text
                    corrected_text = line_ocr_result.get('full_corrected_text', line_ocr_result.get('full_raw_text', regular_ocr_text))
                    
                    # Save inference data with line segments
                    inference_data = create_inference_data(
                        filename=filename,
                        original_text=regular_ocr_text,
                        corrected_text=corrected_text,
                        line_segments=line_ocr_result['line_segments'],
                        total_lines=line_ocr_result['total_lines'],
                        pipeline=line_ocr_result.get('pipeline', 'unknown'),
                        gemini_processing=line_ocr_result.get('gemini_processing', False)
                    )
                else:
                    # Fallback to regular OCR
                    regular_ocr_text = ocr_processing_service.ocr_models.perform_ocr_inference(file_path)
                    inference_data = create_inference_data(
                        filename=filename,
                        original_text=regular_ocr_text,
                        corrected_text=regular_ocr_text,
                        line_segments=[],
                        total_lines=0,
                        pipeline='fallback',
                        gemini_processing=False
                    )

                inference_path = os.path.join(config.inference_folder, f"{filename}.json")
                save_json_data(inference_data, inference_path)

                return jsonify({
                    'success': True, 
                    'filename': filename, 
                    'inference': inference_data['original_text'],
                    'corrected_text': inference_data['corrected_text'],
                    'line_segments': inference_data['line_segments'],
                    'total_lines': inference_data['total_lines'],
                    'pipeline': inference_data['pipeline'],
                    'gemini_processing': inference_data['gemini_processing']
                })

        except Exception as e:
            print("Upload error:", str(e))
            return jsonify({'success': False, 'error': str(e)}), 500

    @routes_bp.route('/image/<filename>')
    def get_image(filename):
        """Serve images from upload folder"""
        try:
            return send_from_directory(config.upload_folder, filename)
        except Exception as e:
            print(f"Error serving image {filename}: {e}")
            return jsonify({'error': 'Image not found'}), 404

    @routes_bp.route('/line_segment/<filename>')
    def get_line_segment(filename):
        """Serve line segment images"""
        try:
            return send_from_directory(config.line_segments_folder, filename)
        except Exception as e:
            print(f"Error serving line segment {filename}: {e}")
            return jsonify({'error': 'Line segment not found'}), 404

    @routes_bp.route('/get_current_image')
    def get_current_image():
        """Get current image name"""
        images = [f for f in os.listdir(config.upload_folder) if is_image_file(f, config.image_extensions)]
        
        if images and 0 <= ocr_processing_service.current_image_index < len(images):
            return jsonify({'image_name': images[ocr_processing_service.current_image_index]})
        return jsonify({'image_name': None})

    @routes_bp.route('/next_image', methods=['POST'])
    def next_image():
        """Move to next image"""
        images = [f for f in os.listdir(config.upload_folder) if is_image_file(f, config.image_extensions)]
        
        if images:
            ocr_processing_service.current_image_index = (ocr_processing_service.current_image_index + 1) % len(images)
            return jsonify({'image_name': images[ocr_processing_service.current_image_index]})
        else:
            return jsonify({'image_name': None})

    @routes_bp.route('/get_inference/<filename>')
    def get_inference(filename):
        """Get inference data for a specific image"""
        try:
            inference_path = os.path.join(config.inference_folder, f"{filename}.json")
            data = load_json_data(inference_path)
            if data:
                return jsonify(data)
            else:
                return jsonify({'error': 'No inference found', 'image': filename})
        except Exception as e:
            print(f"Error loading inference for {filename}: {e}")
            return jsonify({'error': f'Error loading inference: {str(e)}'})

    @routes_bp.route('/update_inference', methods=['POST'])
    def update_inference():
        """Update corrected text for an image"""
        try:
            data = request.json
            filename = data.get('image')
            corrected_text = data.get('corrected_text')
            
            if not filename:
                return jsonify({'success': False, 'error': 'No filename provided'})
            
            inference_path = os.path.join(config.inference_folder, f"{filename}.json")
            inference_data = load_json_data(inference_path)
            
            if inference_data:
                inference_data['corrected_text'] = corrected_text
                inference_data['last_updated'] = time.time()
                save_json_data(inference_data, inference_path)
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Inference file not found'})
        
        except Exception as e:
            print(f"Error updating inference: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/update_line_ocr', methods=['POST'])
    def update_line_ocr():
        """Update OCR text for a specific line"""
        try:
            data = request.json
            filename = data.get('image')
            line_index = data.get('line_index')
            corrected_text = data.get('corrected_text')
            
            if not filename or line_index is None:
                return jsonify({'success': False, 'error': 'Missing filename or line_index'})
            
            inference_path = os.path.join(config.inference_folder, f"{filename}.json")
            inference_data = load_json_data(inference_path)
            
            if inference_data:
                # Update the specific line's OCR text
                if 'line_segments' in inference_data:
                    for segment in inference_data['line_segments']:
                        if segment['line_index'] == line_index:
                            segment['ocr_text'] = corrected_text
                            # Keep a parallel corrected field if present in pipeline
                            segment['ocr_text_corrected'] = corrected_text
                            break
                    
                    # Recompute combined corrected text for convenience
                    try:
                        combined_text = combine_text_from_segments(inference_data['line_segments'])
                        inference_data['corrected_text'] = combined_text
                    except Exception:
                        pass
                
                inference_data['last_updated'] = time.time()
                save_json_data(inference_data, inference_path)
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Inference file not found'})
        
        except Exception as e:
            print(f"Error updating line OCR: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/rerun_inference', methods=['POST'])
    def rerun_inference():
        """Re-run OCR inference on an image"""
        try:
            data = request.json
            filename = data.get('image')
            
            if not filename:
                return jsonify({'success': False, 'error': 'No filename provided'})
            
            image_path = os.path.join(config.upload_folder, filename)
            if not os.path.exists(image_path):
                return jsonify({'success': False, 'error': 'Image file not found'})
            
            # Check if client requests split processing on rerun
            split_image = bool(data.get('split_image', False))
            
            if split_image:
                print(f"Re-running split processing for {filename}...")
                split_result = ocr_processing_service.process_split_image(image_path, filename)
                if not split_result.get('success'):
                    return jsonify({'success': False, 'error': split_result.get('error', 'Split processing failed')})
                
                # Load the just-saved inference for response consistency
                inference_path = os.path.join(config.inference_folder, f"{filename}.json")
                saved = load_json_data(inference_path)
                if saved:
                    response_payload = {
                        'success': True,
                        'inference': saved.get('original_text', saved.get('combined_text', '')),
                        'corrected_text': saved.get('corrected_text', saved.get('combined_text', '')),
                        'line_segments': saved.get('line_segments', []),
                        'total_lines': saved.get('total_lines', 0),
                        'pipeline': saved.get('pipeline', 'split_processing'),
                        'gemini_processing': saved.get('gemini_processing', False),
                        'split_processing': True
                    }
                    return jsonify(response_payload)
            
            # Perform line segmentation and OCR
            print(f"Re-running line segmentation and OCR for {filename}...")
            line_ocr_result = ocr_processing_service.perform_line_segmentation_ocr(image_path)
            
            # Also perform regular OCR
            regular_ocr_text = ocr_processing_service.ocr_models.perform_ocr_inference(image_path)
            
            # Use LLM-corrected text if available
            corrected_text = line_ocr_result.get('full_corrected_text', line_ocr_result.get('full_raw_text', regular_ocr_text))
            
            # Update inference file
            inference_data = create_inference_data(
                filename=filename,
                original_text=regular_ocr_text,
                corrected_text=corrected_text,
                line_segments=line_ocr_result.get('line_segments', []),
                total_lines=line_ocr_result.get('total_lines', 0),
                pipeline=line_ocr_result.get('pipeline', 'unknown'),
                gemini_processing=line_ocr_result.get('gemini_processing', False),
                rerun_count=1
            )
            
            # Check if file exists and increment rerun count
            inference_path = os.path.join(config.inference_folder, f"{filename}.json")
            existing_data = load_json_data(inference_path)
            if existing_data:
                inference_data['rerun_count'] = existing_data.get('rerun_count', 0) + 1
            
            save_json_data(inference_data, inference_path)
            
            return jsonify({
                'success': True, 
                'inference': regular_ocr_text,
                'corrected_text': corrected_text,
                'line_segments': line_ocr_result.get('line_segments', []),
                'total_lines': line_ocr_result.get('total_lines', 0),
                'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                'gemini_processing': line_ocr_result.get('gemini_processing', False),
                'split_processing': False
            })
        
        except Exception as e:
            print(f"Error re-running inference: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/apply_gemini_correction', methods=['POST'])
    def apply_gemini_correction():
        """Apply LLM correction to existing OCR results"""
        try:
            data = request.json
            filename = data.get('image')
            
            if not filename:
                return jsonify({'success': False, 'error': 'No filename provided'})
            
            # Load existing inference data
            inference_path = os.path.join(config.inference_folder, f"{filename}.json")
            inference_data = load_json_data(inference_path)
            
            if not inference_data:
                return jsonify({'success': False, 'error': 'No inference data found'})
            
            # Get line segments
            line_segments = inference_data.get('line_segments', [])
            if not line_segments:
                return jsonify({'success': False, 'error': 'No line segments found'})
            
            # Apply LLM correction
            print(f"Applying LLM correction to {len(line_segments)} line segments...")
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
            inference_data['gemini_processing'] = True
            inference_data['gemini_correction_timestamp'] = time.time()
            
            # Save updated data
            save_json_data(inference_data, inference_path)
            
            return jsonify({
                'success': True,
                'corrected_text': full_corrected_text,
                'line_segments': corrected_segments,
                'total_lines': len(corrected_segments),
                'gemini_processing': True
            })
        
        except Exception as e:
            print(f"Error applying LLM correction: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/save', methods=['POST'])
    def save_annotations():
        """Save annotation data"""
        try:
            data = request.json
            if not data or 'image' not in data:
                return jsonify({'success': False, 'error': 'Invalid data'})
            
            # Add timestamp
            data['saved_at'] = time.time()
            
            annotation_path = os.path.join(config.annotation_folder, f"{data['image']}.json")
            save_json_data(data, annotation_path)
            
            print(f"Annotations saved for {data['image']}: {len(data.get('annotations', []))} boxes")
            return jsonify({'success': True})
        
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/apply_gemini_to_pdf', methods=['POST'])
    def apply_gemini_to_pdf():
        """Apply LLM correction to all pages of an existing PDF"""
        try:
            data = request.json
            pdf_filename = data.get('pdf_filename')
            
            if not pdf_filename:
                return jsonify({'success': False, 'error': 'No PDF filename provided'})
            
            # Load PDF manifest to get page filenames
            base_name, _ = os.path.splitext(pdf_filename)
            manifest_path = os.path.join(config.inference_folder, f"{base_name}_manifest.json")
            manifest = load_json_data(manifest_path)
            
            if not manifest:
                return jsonify({'success': False, 'error': 'PDF manifest not found'})
            
            page_filenames = manifest.get('page_filenames', [])
            if not page_filenames:
                return jsonify({'success': False, 'error': 'No pages found in manifest'})
            
            # Apply LLM correction to all pages
            print(f"Applying LLM correction to {len(page_filenames)} pages of {pdf_filename}...")
            previous_context = ""
            processed_pages = 0
            
            for i, page_filename in enumerate(page_filenames):
                print(f"Processing page {i + 1}/{len(page_filenames)}: {page_filename}")
                
                inference_path = os.path.join(config.inference_folder, f"{page_filename}.json")
                inference_data = load_json_data(inference_path)
                
                if not inference_data:
                    print(f"Warning: Inference file not found for {page_filename}")
                    continue
                
                original_text = inference_data.get('original_text', '')
                
                if original_text.strip():
                    # Apply LLM correction with context from previous page
                    corrected_text, status = llm_service.process_text_with_gemini(original_text, previous_context)
                    
                    # Update inference data with LLM results
                    inference_data['corrected_text'] = corrected_text
                    inference_data['gemini_processing'] = (status == "success")
                    inference_data['gemini_correction_status'] = status
                    inference_data['gemini_correction_timestamp'] = time.time()
                    inference_data['gemini_reprocessed'] = True
                    
                    # Update context for next page
                    if status == "success":
                        previous_context = get_last_two_lines(corrected_text)
                    else:
                        previous_context = get_last_two_lines(original_text)
                    
                    processed_pages += 1
                else:
                    # Empty text - mark as processed
                    inference_data['gemini_processing'] = True
                    inference_data['gemini_correction_status'] = "empty_text"
                    inference_data['gemini_correction_timestamp'] = time.time()
                    inference_data['gemini_reprocessed'] = True
                
                # Save updated inference data
                save_json_data(inference_data, inference_path)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            # Update manifest
            manifest['gemini_processing_completed'] = True
            manifest['gemini_reprocessed_at'] = time.time()
            manifest['processed_pages_count'] = processed_pages
            
            save_json_data(manifest, manifest_path)
            
            return jsonify({
                'success': True,
                'pdf_filename': pdf_filename,
                'total_pages': len(page_filenames),
                'processed_pages': processed_pages,
                'gemini_processing_completed': True
            })
        
        except Exception as e:
            print(f"Error applying LLM to PDF: {e}")
            return jsonify({'success': False, 'error': str(e)})

    @routes_bp.route('/health')
    def health_check():
        """Enhanced health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': (ocr_processing_service.ocr_models.processor is not None and 
                           ocr_processing_service.ocr_models.model is not None),
            'textline_model_loaded': ocr_processing_service.ocr_models.textline_extractor is not None,
            'advanced_pipeline_available': ocr_processing_service.ocr_models.textline_extractor is not None,
            'gemini_api_available': hasattr(llm_service, 'config') and bool(llm_service.config.gemini_api_key),
            'upload_folder': config.upload_folder,
            'folders_exist': {
                'uploads': os.path.exists(config.upload_folder),
                'annotations': os.path.exists(config.annotation_folder),
                'inferences': os.path.exists(config.inference_folder),
                'line_segments': os.path.exists(config.line_segments_folder)
            }
        })

    @routes_bp.route('/llm_health')
    def llm_health():
        """Health check for LLM backends"""
        
        # Check LLM availability
        gemini_available = hasattr(llm_service, 'config') and bool(llm_service.config.gemini_api_key)
        hf_available = hasattr(llm_service, 'config') and bool(llm_service.config.hf_api_token)
        # Replace the local_llama_available line with this
        try:
            import requests as req
            ollama_response = req.get("http://localhost:11434", timeout=3)
            local_llama_available = "Ollama is running" in ollama_response.text
        except Exception:
            local_llama_available = False
        
        # Check if Llama license acceptance is required
        llama_license_required = True  # Always true since it's a gated model
        
        notes = []
        if not gemini_available:
            notes.append("Gemini: Missing GEMINI_API_KEY")
        if not hf_available:
            notes.append("HuggingFace: Missing HF_API_TOKEN") 
        if not local_llama_available:
            if not hasattr(llm_service, 'config') or not llm_service.config.llama_model_path:
                notes.append("LLaMA: Missing LLAMA_MODEL_PATH")
            else:
                notes.append("LLaMA: Model file not found")
        
        return jsonify({
            "gemini": gemini_available,
            "hf": hf_available, 
            "local_llama": local_llama_available,
            "llama_license_required": llama_license_required,
            "llama_license_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
            "notes": "; ".join(notes) if notes else "All available backends ready"
        })

    return routes_bp
