"""
Configuration management for Flask OCR Application
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    trocr_spanish_model: str = "qantev/trocr-large-spanish"
    trocr_fallback_model: str = "microsoft/trocr-base-printed"
    textline_model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class LLMConfig:
    """Configuration for LLM services"""
    gemini_api_key: Optional[str] = None
    hf_api_token: Optional[str] = None
    llama_model_path: Optional[str] = None
    fallback_chunk_size: int = 8000
    llm_timeout_seconds: int = 15
    max_retries: int = 2


@dataclass
class FileConfig:
    """Configuration for file handling"""
    upload_folder: str = "uploads"
    annotation_folder: str = "annotations"
    inference_folder: str = "inferences"
    line_segments_folder: str = "line_segments"
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = None
    image_extensions: set = None

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
        if self.image_extensions is None:
            self.image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


@dataclass
class AppConfig:
    """Main application configuration"""
    # Core settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Model configuration
    models: ModelConfig = None
    
    # LLM configuration
    llm: LLMConfig = None
    
    # File configuration
    files: FileConfig = None
    
    # Logging configuration
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.models is None:
            self.models = ModelConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.files is None:
            self.files = FileConfig()


def load_environment_variables() -> Dict[str, Any]:
    """Load and validate environment variables"""
    env_vars = {}
    
    # Core application settings
    env_vars['debug'] = os.getenv('DEBUG', 'false').lower() == 'true'
    env_vars['host'] = os.getenv('HOST', '0.0.0.0')
    env_vars['port'] = int(os.getenv('PORT', '5000'))
    env_vars['log_level'] = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Model configuration
    env_vars['textline_model_path'] = os.getenv('TEXTLINE_MODEL_PATH')
    env_vars['device'] = os.getenv('DEVICE', 'auto')
    
    # LLM configuration
    env_vars['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
    env_vars['hf_api_token'] = os.getenv('HF_API_TOKEN')
    env_vars['llama_model_path'] = os.getenv('LLAMA_MODEL_PATH')
    env_vars['fallback_chunk_size'] = int(os.getenv('FALLBACK_CHUNK_SIZE', '8000'))
    env_vars['llm_timeout_seconds'] = int(os.getenv('LLM_TIMEOUT_SECONDS', '15'))
    env_vars['max_retries'] = int(os.getenv('MAX_RETRIES', '2'))
    
    # File configuration
    env_vars['upload_folder'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    env_vars['annotation_folder'] = os.getenv('ANNOTATION_FOLDER', 'annotations')
    env_vars['inference_folder'] = os.getenv('INFERENCE_FOLDER', 'inferences')
    env_vars['line_segments_folder'] = os.getenv('LINE_SEGMENTS_FOLDER', 'line_segments')
    env_vars['max_file_size'] = int(os.getenv('MAX_FILE_SIZE', str(16 * 1024 * 1024)))
    
    return env_vars


def validate_config(config: AppConfig) -> None:
    """Validate configuration and create necessary directories"""
    errors = []
    
    # Validate required directories
    required_dirs = [
        config.files.upload_folder,
        config.files.annotation_folder,
        config.files.inference_folder,
        config.files.line_segments_folder
    ]
    
    for directory in required_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            errors.append(f"Cannot create directory {directory}: {e}")
    
    # Validate model paths if provided
    if config.models.textline_model_path and not os.path.exists(config.models.textline_model_path):
        errors.append(f"Textline model path does not exist: {config.models.textline_model_path}")
    
    if config.llm.llama_model_path and not os.path.exists(config.llm.llama_model_path):
        errors.append(f"LLaMA model path does not exist: {config.llm.llama_model_path}")
    
    # Validate file size
    if config.files.max_file_size <= 0:
        errors.append("Max file size must be positive")
    
    # Validate chunk size
    if config.llm.fallback_chunk_size <= 0:
        errors.append("Fallback chunk size must be positive")
    
    # Validate timeout
    if config.llm.llm_timeout_seconds <= 0:
        errors.append("LLM timeout must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


def get_config() -> AppConfig:
    """Get application configuration from environment variables"""
    env_vars = load_environment_variables()
    
    # Create model configuration
    model_config = ModelConfig(
        textline_model_path=env_vars.get('textline_model_path'),
        device=env_vars.get('device', 'auto')
    )
    
    # Create LLM configuration
    llm_config = LLMConfig(
        gemini_api_key=env_vars.get('gemini_api_key'),
        hf_api_token=env_vars.get('hf_api_token'),
        llama_model_path=env_vars.get('llama_model_path'),
        fallback_chunk_size=env_vars.get('fallback_chunk_size', 8000),
        llm_timeout_seconds=env_vars.get('llm_timeout_seconds', 15),
        max_retries=env_vars.get('max_retries', 2)
    )
    
    # Create file configuration
    file_config = FileConfig(
        upload_folder=env_vars.get('upload_folder', 'uploads'),
        annotation_folder=env_vars.get('annotation_folder', 'annotations'),
        inference_folder=env_vars.get('inference_folder', 'inferences'),
        line_segments_folder=env_vars.get('line_segments_folder', 'line_segments'),
        max_file_size=env_vars.get('max_file_size', 16 * 1024 * 1024)
    )
    
    # Create main configuration
    config = AppConfig(
        debug=env_vars.get('debug', False),
        host=env_vars.get('host', '0.0.0.0'),
        port=env_vars.get('port', 5000),
        log_level=env_vars.get('log_level', 'INFO'),
        models=model_config,
        llm=llm_config,
        files=file_config
    )
    
    # Validate configuration
    validate_config(config)
    
    return config


def setup_logging(config: AppConfig) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    
    logger.info(f"Logging configured at level: {config.log_level}")


def print_config_summary(config: AppConfig) -> None:
    """Print configuration summary for debugging"""
    logger.info("=== Configuration Summary ===")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Log level: {config.log_level}")
    logger.info(f"Upload folder: {config.files.upload_folder}")
    logger.info(f"Max file size: {config.files.max_file_size / (1024*1024):.1f}MB")
    logger.info(f"Gemini API: {'✓' if config.llm.gemini_api_key else '✗'}")
    logger.info(f"HuggingFace API: {'✓' if config.llm.hf_api_token else '✗'}")
    logger.info(f"LLaMA model: {'✓' if config.llm.llama_model_path else '✗'}")
    logger.info(f"Textline model: {'✓' if config.models.textline_model_path else '✗'}")
    logger.info("=============================")
