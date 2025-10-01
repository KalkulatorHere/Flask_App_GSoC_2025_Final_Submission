# Modular OCR Flask Application

This is a clean, modular version of the OCR Flask application with separated concerns and improved maintainability.

## Structure

```
modular_app/
├── __init__.py              # Package initialization
├── config.py               # Configuration management
├── llm_service.py          # LLM services (Gemini, HuggingFace, LLaMA)
├── ocr_models.py           # OCR models (TrOCR, Detectron2)
├── ocr_processing.py       # Main OCR processing pipeline
├── routes.py               # Flask routes and API endpoints
├── utils.py                # Utility functions
├── index.html              # Web interface
└── README.md               # This file
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Centralized configuration
- **LLM Integration**: Multi-backend LLM support with fallback
- **OCR Processing**: Advanced line segmentation and text recognition
- **File Handling**: PDF processing, image splitting, utilities
- **API Endpoints**: Complete REST API for all functionality

## Usage

Run the application using the main launcher:

```bash
python app_final.py
```

The application will be available at `http://localhost:5000`

## Services

### LLMService
- Handles text correction using multiple backends
- Supports Gemini, HuggingFace, and Local LLaMA
- Automatic fallback system
- Text chunking for large inputs

### OCRModelsService
- Manages TrOCR and Detectron2 models
- Advanced textline extraction
- Fallback OCR processing
- Model loading and initialization

### OCRProcessingService
- Main OCR processing pipeline
- Image splitting functionality
- PDF processing
- Line segmentation and text recognition

## Configuration

All configuration is managed through `config.py`:
- API keys and tokens
- Directory paths
- Model paths
- Processing settings

## Dependencies

The modular application uses the same dependencies as the original:
- Flask and Flask-CORS
- Transformers (TrOCR)
- Detectron2 (textline detection)
- OpenCV (image processing)
- PyMuPDF (PDF processing)
- Google Generative AI (Gemini)
- llama-cpp-python (Local LLaMA)

## Environment Variables

Set these environment variables for full functionality:
- `GEMINI_API_KEY`: Google Gemini API key
- `HF_API_TOKEN`: HuggingFace API token
- `LLAMA_MODEL_PATH`: Path to local LLaMA model
- `FALLBACK_CHUNK_SIZE`: Text chunking size (default: 8000)
- `LLM_TIMEOUT_SECONDS`: LLM timeout (default: 15)

