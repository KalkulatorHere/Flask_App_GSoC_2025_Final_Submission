"""
Validation utilities for Flask OCR Application
"""

import logging
import os
from typing import List, Dict, Optional, Any
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)


class Validators:
    """Utility class for input validation"""
    
    @staticmethod
    def validate_file_upload(file: FileStorage, allowed_extensions: set, max_size: int) -> Dict[str, Any]:
        """Validate uploaded file"""
        errors = []
        
        if not file:
            errors.append("No file provided")
            return {'valid': False, 'errors': errors}
        
        if not file.filename:
            errors.append("No filename provided")
            return {'valid': False, 'errors': errors}
        
        # Check file extension
        if '.' not in file.filename:
            errors.append("File has no extension")
        else:
            extension = file.filename.rsplit('.', 1)[1].lower()
            if extension not in allowed_extensions:
                errors.append(f"File extension '{extension}' not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            file_size_mb = file_size / (1024 * 1024)
            errors.append(f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb:.1f}MB)")
        
        if file_size == 0:
            errors.append("File is empty")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'file_size': file_size,
            'filename': file.filename
        }
    
    @staticmethod
    def validate_image_file(file_path: str) -> Dict[str, Any]:
        """Validate image file"""
        errors = []
        
        if not os.path.exists(file_path):
            errors.append("File does not exist")
            return {'valid': False, 'errors': errors}
        
        if not os.path.isfile(file_path):
            errors.append("Path is not a file")
            return {'valid': False, 'errors': errors}
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            errors.append("File is empty")
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            errors.append("File is not readable")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'file_size': file_size
        }
    
    @staticmethod
    def validate_text_input(text: str, min_length: int = 1, max_length: int = None) -> Dict[str, Any]:
        """Validate text input"""
        errors = []
        
        if not text:
            errors.append("Text is empty")
            return {'valid': False, 'errors': errors}
        
        if len(text) < min_length:
            errors.append(f"Text too short (minimum: {min_length} characters)")
        
        if max_length and len(text) > max_length:
            errors.append(f"Text too long (maximum: {max_length} characters)")
        
        # Check for valid characters (basic check)
        if not isinstance(text, str):
            errors.append("Text must be a string")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'length': len(text)
        }
    
    @staticmethod
    def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['upload_folder', 'inference_folder', 'max_file_size']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required configuration field: {field}")
        
        # Validate upload folder
        if 'upload_folder' in config:
            upload_folder = config['upload_folder']
            if not os.path.exists(upload_folder):
                try:
                    os.makedirs(upload_folder, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create upload folder: {e}")
            elif not os.access(upload_folder, os.W_OK):
                errors.append("Upload folder is not writable")
        
        # Validate file size
        if 'max_file_size' in config:
            max_size = config['max_file_size']
            if not isinstance(max_size, int) or max_size <= 0:
                errors.append("max_file_size must be a positive integer")
            elif max_size > 100 * 1024 * 1024:  # 100MB
                warnings.append("max_file_size is very large (>100MB)")
        
        # Validate API keys (optional but warn if missing)
        api_keys = ['gemini_api_key', 'hf_api_token', 'llama_model_path']
        for key in api_keys:
            if key in config and not config[key]:
                warnings.append(f"{key} is not configured")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_inference_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inference data structure"""
        errors = []
        
        required_fields = ['image', 'timestamp']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate timestamp
        if 'timestamp' in data:
            timestamp = data['timestamp']
            if not isinstance(timestamp, (int, float)):
                errors.append("timestamp must be a number")
            elif timestamp <= 0:
                errors.append("timestamp must be positive")
        
        # Validate line segments if present
        if 'line_segments' in data:
            line_segments = data['line_segments']
            if not isinstance(line_segments, list):
                errors.append("line_segments must be a list")
            else:
                for i, segment in enumerate(line_segments):
                    if not isinstance(segment, dict):
                        errors.append(f"line_segments[{i}] must be a dictionary")
                    else:
                        required_segment_fields = ['line_index', 'ocr_text']
                        for field in required_segment_fields:
                            if field not in segment:
                                errors.append(f"line_segments[{i}] missing required field: {field}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    @staticmethod
    def validate_api_request(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
        """Validate API request data"""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Request data must be a JSON object")
            return {'valid': False, 'errors': errors}
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None:
                errors.append(f"Field '{field}' cannot be null")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "unnamed_file"
        
        # Remove path separators and other dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty
        if not filename:
            filename = "unnamed_file"
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def validate_model_path(model_path: str) -> Dict[str, Any]:
        """Validate model file path"""
        errors = []
        
        if not model_path:
            errors.append("Model path is empty")
            return {'valid': False, 'errors': errors}
        
        if not os.path.exists(model_path):
            errors.append(f"Model file does not exist: {model_path}")
        
        if not os.path.isfile(model_path):
            errors.append(f"Model path is not a file: {model_path}")
        
        if not os.access(model_path, os.R_OK):
            errors.append(f"Model file is not readable: {model_path}")
        
        # Check file extension
        if not model_path.endswith(('.pth', '.pt', '.bin', '.safetensors')):
            errors.append("Model file should have a valid extension (.pth, .pt, .bin, .safetensors)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    @staticmethod
    def validate_health_check_response(response: Dict[str, Any]) -> bool:
        """Validate health check response format"""
        required_fields = ['status']
        return all(field in response for field in required_fields)
