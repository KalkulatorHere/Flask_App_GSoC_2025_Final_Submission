"""
Custom exception classes for Flask OCR Application
"""


class OCRException(Exception):
    """Base exception for OCR-related errors"""
    pass


class ModelLoadException(OCRException):
    """Exception raised when model loading fails"""
    pass


class TextlineDetectionException(OCRException):
    """Exception raised when textline detection fails"""
    pass


class LLMServiceException(Exception):
    """Exception raised when LLM service operations fail"""
    pass


class FileProcessingException(Exception):
    """Exception raised when file processing fails"""
    pass


class PDFProcessingException(FileProcessingException):
    """Exception raised when PDF processing fails"""
    pass


class ImageProcessingException(FileProcessingException):
    """Exception raised when image processing fails"""
    pass


class ValidationException(Exception):
    """Exception raised when input validation fails"""
    pass


class ConfigurationException(Exception):
    """Exception raised when configuration is invalid"""
    pass
