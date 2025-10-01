"""
API module for Flask OCR Application
"""

from .routes import create_api_routes
from .handlers import RequestHandler, ResponseHandler

__all__ = ['create_api_routes', 'RequestHandler', 'ResponseHandler']
