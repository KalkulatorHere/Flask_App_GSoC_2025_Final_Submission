"""
Request and response handlers for Flask OCR Application
"""

import logging
from typing import Dict, Any, Optional, Tuple
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

from ..exceptions import ValidationException, FileProcessingException
from ..utils import Validators
from ..config import AppConfig

logger = logging.getLogger(__name__)


class RequestHandler:
    """Handles request processing and validation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.validators = Validators()
    
    def validate_file_upload(self, file: FileStorage) -> Dict[str, Any]:
        """Validate uploaded file"""
        validation_result = self.validators.validate_file_upload(
            file, 
            self.config.files.allowed_extensions,
            self.config.files.max_file_size
        )
        
        if not validation_result['valid']:
            raise ValidationException(f"File validation failed: {', '.join(validation_result['errors'])}")
        
        return validation_result
    
    def validate_json_request(self, required_fields: list) -> Dict[str, Any]:
        """Validate JSON request data"""
        if not request.is_json:
            raise ValidationException("Request must be JSON")
        
        data = request.get_json()
        validation_result = self.validators.validate_api_request(data, required_fields)
        
        if not validation_result['valid']:
            raise ValidationException(f"Request validation failed: {', '.join(validation_result['errors'])}")
        
        return data
    
    def get_file_from_request(self) -> Tuple[FileStorage, str]:
        """Extract and validate file from request"""
        if 'file' not in request.files:
            raise ValidationException("No file part in request")
        
        file = request.files['file']
        if file.filename == '':
            raise ValidationException("No selected file")
        
        # Validate file
        self.validate_file_upload(file)
        
        return file, file.filename
    
    def get_form_data(self) -> Dict[str, Any]:
        """Extract form data from request"""
        return {
            'split_image': request.form.get('split_image', 'false').lower() == 'true',
            'skip_llm': request.form.get('skip_llm', 'false').lower() == 'true'
        }
    
    def get_query_params(self) -> Dict[str, Any]:
        """Extract query parameters from request"""
        return {
            'page': request.args.get('page', 1, type=int),
            'limit': request.args.get('limit', 10, type=int),
            'format': request.args.get('format', 'json')
        }


class ResponseHandler:
    """Handles response formatting and error handling"""
    
    @staticmethod
    def success_response(data: Dict[str, Any], message: str = "Success", status_code: int = 200) -> Tuple[Dict[str, Any], int]:
        """Create success response"""
        response = {
            'success': True,
            'message': message,
            'data': data
        }
        return jsonify(response), status_code
    
    @staticmethod
    def error_response(message: str, error_code: str = "ERROR", status_code: int = 400, details: Optional[Dict] = None) -> Tuple[Dict[str, Any], int]:
        """Create error response"""
        response = {
            'success': False,
            'error': message,
            'error_code': error_code,
            'status_code': status_code
        }
        
        if details:
            response['details'] = details
        
        return jsonify(response), status_code
    
    @staticmethod
    def validation_error_response(errors: list, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
        """Create validation error response"""
        return ResponseHandler.error_response(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            status_code=status_code,
            details={'errors': errors}
        )
    
    @staticmethod
    def file_error_response(message: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
        """Create file processing error response"""
        return ResponseHandler.error_response(
            message=message,
            error_code="FILE_ERROR",
            status_code=status_code
        )
    
    @staticmethod
    def model_error_response(message: str, status_code: int = 500) -> Tuple[Dict[str, Any], int]:
        """Create model error response"""
        return ResponseHandler.error_response(
            message=message,
            error_code="MODEL_ERROR",
            status_code=status_code
        )
    
    @staticmethod
    def service_error_response(message: str, status_code: int = 500) -> Tuple[Dict[str, Any], int]:
        """Create service error response"""
        return ResponseHandler.error_response(
            message=message,
            error_code="SERVICE_ERROR",
            status_code=status_code
        )
    
    @staticmethod
    def not_found_response(resource: str) -> Tuple[Dict[str, Any], int]:
        """Create not found response"""
        return ResponseHandler.error_response(
            message=f"{resource} not found",
            error_code="NOT_FOUND",
            status_code=404
        )
    
    @staticmethod
    def method_not_allowed_response(method: str) -> Tuple[Dict[str, Any], int]:
        """Create method not allowed response"""
        return ResponseHandler.error_response(
            message=f"Method {method} not allowed",
            error_code="METHOD_NOT_ALLOWED",
            status_code=405
        )
    
    @staticmethod
    def rate_limit_response() -> Tuple[Dict[str, Any], int]:
        """Create rate limit response"""
        return ResponseHandler.error_response(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429
        )
    
    @staticmethod
    def server_error_response(message: str = "Internal server error") -> Tuple[Dict[str, Any], int]:
        """Create server error response"""
        return ResponseHandler.error_response(
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            status_code=500
        )
    
    @staticmethod
    def health_check_response(services: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Create health check response"""
        overall_health = all(
            service.get('service_ready', False) 
            for service in services.values() 
            if isinstance(service, dict)
        )
        
        response = {
            'status': 'healthy' if overall_health else 'unhealthy',
            'services': services,
            'timestamp': __import__('time').time()
        }
        
        status_code = 200 if overall_health else 503
        return jsonify(response), status_code
