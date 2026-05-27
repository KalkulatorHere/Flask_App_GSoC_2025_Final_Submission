"""
Configuration management for the modular OCR application
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional

# Environment variable configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')
FALLBACK_CHUNK_SIZE = int(os.getenv('FALLBACK_CHUNK_SIZE', '8000'))
LLM_TIMEOUT_SECONDS = int(os.getenv('LLM_TIMEOUT_SECONDS', '600'))

# Directory configuration
UPLOAD_FOLDER = 'uploads'
ANNOTATION_FOLDER = 'annotations'
INFERENCE_FOLDER = 'Model_Outputs_Json'
LINE_SEGMENTS_FOLDER = 'line_segments'

# File type configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Model paths
TEXTLINE_MODEL_PATH = r"C:\Users\rdb104\Documents\caserepos\models\model_final (8) (1).pth"

@dataclass
class AppConfig:
    """Application configuration"""
    # API Keys
    gemini_api_key: Optional[str] = GEMINI_API_KEY
    hf_api_token: Optional[str] = HF_API_TOKEN
    llama_model_path: Optional[str] = LLAMA_MODEL_PATH
    
    # Processing settings
    fallback_chunk_size: int = FALLBACK_CHUNK_SIZE
    llm_timeout_seconds: int = LLM_TIMEOUT_SECONDS
    
    # Directory settings
    upload_folder: str = UPLOAD_FOLDER
    annotation_folder: str = ANNOTATION_FOLDER
    inference_folder: str = INFERENCE_FOLDER
    line_segments_folder: str = LINE_SEGMENTS_FOLDER
    
    # File settings
    allowed_extensions: set = None
    image_extensions: set = None
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    
    # Model settings
    textline_model_path: str = TEXTLINE_MODEL_PATH
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ALLOWED_EXTENSIONS
        if self.image_extensions is None:
            self.image_extensions = IMAGE_EXTENSIONS

def get_config() -> AppConfig:
    """Get application configuration"""
    return AppConfig()

def setup_directories(config: AppConfig) -> None:
    """Create necessary directories"""
    directories = [
        config.upload_folder,
        config.annotation_folder,
        config.inference_folder,
        config.line_segments_folder
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created/verified directory: {directory}")

def print_config_summary(config: AppConfig) -> None:
    """Print configuration summary"""
    print("=" * 60)
    print("OCR APPLICATION CONFIGURATION")
    print("=" * 60)
    print(f"Gemini API Key: {'✓' if config.gemini_api_key else '✗'}")
    print(f"HuggingFace API Token: {'✓' if config.hf_api_token else '✗'}")
    print(f"LLaMA Model Path: {'✓' if config.llama_model_path else '✗'}")
    print(f"Textline Model Path: {'✓' if os.path.exists(config.textline_model_path) else '✗'}")
    
    try:
        import importlib.metadata
        kraken_version = importlib.metadata.version('kraken')
        print(f"Kraken Segmentation: ✓ ({kraken_version})")
    except Exception:
        print(f"Kraken Segmentation: ✗ (not installed)")

    print(f"Upload Folder: {config.upload_folder}")
    print(f"Max File Size: {config.max_file_size / (1024*1024):.1f}MB")
    print(f"Chunk Size: {config.fallback_chunk_size}")
    print(f"LLM Timeout: {config.llm_timeout_seconds}s")
    print("=" * 60)
