# Advanced OCR Flask Application with Gemini API Integration

This Flask application provides advanced OCR capabilities with line segmentation, TrOCR processing, and Gemini API text correction for historical Spanish documents.

## Features

- **Advanced Line Segmentation**: Uses Detectron2-based textline detection with dynamic padding
- **TrOCR Processing**: Microsoft TrOCR for accurate text recognition
- **Gemini API Integration**: Google Gemini 2.5 Pro for text correction and post-processing
- **Reading Order Detection**: Automatic sorting of textlines in proper reading order
- **RESTful API**: Full API endpoints for upload, processing, and correction
- **Real-time Processing**: Immediate feedback with progress indicators

## Installation

Detectron2 and PyTorch require platform/CUDA-specific wheels. Follow these steps exactly to avoid import errors.

1) Create and activate a fresh virtual environment (Windows PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

2) Install PyTorch for your platform/CUDA:
- Visit `https://pytorch.org/get-started/locally/` and copy the command shown.
- Examples:
  - CUDA 12.1: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
  - CPU-only: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`

3) Install base dependencies from this repo:
```bash
pip install -r requirements.txt
```

4) Install Detectron2 matching your PyTorch and CUDA:
- Find wheel links at `https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.4/index.html` (adjust cu/torch versions).
- Or use CPU wheel if available: `https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.4/index.html`.
- Example (CUDA 12.1, Torch 2.4):
```bash
pip install -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.4/index.html detectron2==0.6
```

If you use a different Torch/CUDA, pick the corresponding wheel folder. A mismatch will cause `ModuleNotFoundError: detectron2` or CUDA errors.

5) Optional: Local LLaMA
- To enable local LLaMA correction, install `llama-cpp-python` (CPU wheels commonly available):
```bash
pip install llama-cpp-python
```
If building from source fails on Windows, leave it uninstalled; the app will run without local LLaMA.

6) Environment variables:
- `GEMINI_API_KEY`: Google Gemini API key (required for Gemini correction)
- `HF_API_TOKEN`: HuggingFace API token (optional)
- `LLAMA_MODEL_PATH`: Path to local LLaMA model (optional)

7) Detectron2 weights/model path:
- Place your Detectron2 weights file where configured in `modular_app/config.py` (`textline_model_path`).
- If the file is missing, the app falls back to a mock textline detector.

## Usage

Quick start (Windows PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app_final.py
```

1. Start the Flask server:
```bash
python app_final.py
```

2. Access the web interface at `http://localhost:5000`

3. Upload images or PDFs for processing

## API Endpoints

### Core Endpoints
- `POST /upload` - Upload and process images
- `GET /image/<filename>` - Retrieve uploaded images
- `GET /line_segment/<filename>` - Retrieve line segment images
- `GET /get_inference/<filename>` - Get processing results

### Processing Endpoints
- `POST /rerun_inference` - Re-run OCR on existing images
- `POST /apply_gemini_correction` - Apply Gemini correction to existing results
- `POST /update_inference` - Update corrected text
- `POST /update_line_ocr` - Update specific line OCR results

### Status Endpoints
- `GET /health` - System health and model status
- `GET /get_current_image` - Get current image information

## Processing Pipeline

1. **Image Upload**: Accepts various image formats and PDFs
2. **Line Segmentation**: Advanced Detectron2-based textline detection
3. **Dynamic Padding**: Intelligent padding based on text spacing
4. **Reading Order**: Automatic sorting in proper reading sequence
5. **TrOCR Processing**: High-quality text recognition
6. **Gemini Correction**: AI-powered text correction and enhancement
7. **Result Storage**: Comprehensive JSON output with metadata

## Output Format

The application generates structured JSON output containing:
- Original and corrected text
- Line segment information with bounding boxes
- Processing pipeline metadata
- Gemini API correction status
- Confidence scores and reading order

## Configuration

Key configuration options:
- `GEMINI_API_KEY`: Your Gemini API key
- `UPLOAD_FOLDER`: Directory for uploaded files
- `MAX_CONTENT_LENGTH`: Maximum file size (16MB default)
- Model paths for Detectron2 and TrOCR

## Error Handling

The application includes comprehensive error handling:
- Graceful fallbacks when models are unavailable
- Retry logic for API calls
- Detailed error messages and logging
- Health check endpoints for monitoring

## Dependencies

- Flask 2.3.3
- PyTorch and Transformers
- OpenCV and NumPy
- Detectron2 for object detection
- Google Generative AI for text correction
- PyMuPDF for PDF processing 
