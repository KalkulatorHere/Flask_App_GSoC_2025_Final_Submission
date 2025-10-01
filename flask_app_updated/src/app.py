"""
Main Flask application entry point for OCR Application
"""

import logging
from flask import Flask
from flask_cors import CORS

from .config import get_config, setup_logging, print_config_summary
from .services import OCRService, LLMService, PDFService, ImageService, FileService
from .api import create_api_routes
from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application"""
    try:
        # Load configuration
        config = get_config()
        
        # Setup logging
        setup_logging(config)
        logger.info("Starting Flask OCR Application...")
        
        # Print configuration summary
        print_config_summary(config)
        
        # Create Flask app
        app = Flask(__name__)
        app.config['UPLOAD_FOLDER'] = config.files.upload_folder
        app.config['MAX_CONTENT_LENGTH'] = config.files.max_file_size
        
        # Enable CORS
        CORS(app)
        
        # Initialize services
        logger.info("Initializing services...")
        services = initialize_services(config)
        
        # Register API routes
        logger.info("Registering API routes...")
        api_blueprint = create_api_routes(config, services)
        app.register_blueprint(api_blueprint)
        
        # Add error handlers
        setup_error_handlers(app)
        
        logger.info("Flask application created successfully")
        return app
        
    except ConfigurationException as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create Flask application: {e}")
        raise


def initialize_services(config) -> dict:
    """Initialize all services with dependency injection"""
    try:
        # Initialize core services
        ocr_service = OCRService(config)
        llm_service = LLMService(config.llm)
        file_service = FileService(config)
        
        # Initialize dependent services
        pdf_service = PDFService(config, ocr_service, llm_service)
        image_service = ImageService(config, ocr_service, llm_service)
        
        services = {
            'ocr_service': ocr_service,
            'llm_service': llm_service,
            'pdf_service': pdf_service,
            'image_service': image_service,
            'file_service': file_service
        }
        
        logger.info("All services initialized successfully")
        return services
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


def setup_error_handlers(app: Flask) -> None:
    """Setup error handlers for the Flask application"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {
            'success': False,
            'error': 'Bad request',
            'error_code': 'BAD_REQUEST',
            'status_code': 400
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {
            'success': False,
            'error': 'Not found',
            'error_code': 'NOT_FOUND',
            'status_code': 404
        }, 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return {
            'success': False,
            'error': 'Method not allowed',
            'error_code': 'METHOD_NOT_ALLOWED',
            'status_code': 405
        }, 405
    
    @app.errorhandler(413)
    def file_too_large(error):
        return {
            'success': False,
            'error': 'File too large',
            'error_code': 'FILE_TOO_LARGE',
            'status_code': 413
        }, 413
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {
            'success': False,
            'error': 'Internal server error',
            'error_code': 'INTERNAL_SERVER_ERROR',
            'status_code': 500
        }, 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}")
        return {
            'success': False,
            'error': 'An unexpected error occurred',
            'error_code': 'UNEXPECTED_ERROR',
            'status_code': 500
        }, 500


def run_app():
    """Run the Flask application"""
    try:
        app = create_app()
        
        # Get configuration
        config = get_config()
        
        logger.info(f"Starting Flask app on {config.host}:{config.port}")
        logger.info(f"Debug mode: {config.debug}")
        
        # Run the app
        app.run(
            debug=config.debug,
            host=config.host,
            port=config.port
        )
        
    except Exception as e:
        logger.error(f"Failed to run Flask application: {e}")
        raise


if __name__ == '__main__':
    run_app()
