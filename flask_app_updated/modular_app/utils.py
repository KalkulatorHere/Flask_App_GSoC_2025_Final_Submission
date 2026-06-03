"""
Utility functions for file handling, image processing, and PDF conversion
"""
import os
import time
import cv2
import numpy as np
from typing import List, Dict, Optional
from werkzeug.utils import secure_filename

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def is_image_file(filename: str, image_extensions: set) -> bool:
    """Check if file is an image based on extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def get_image_dimensions(file) -> Dict:
    """Get image dimensions from file object"""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            
            # Read image with OpenCV
            image = cv2.imread(tmp_file.name)
            if image is not None:
                height, width = image.shape[:2]
                os.unlink(tmp_file.name)
                return {'width': width, 'height': height, 'success': True}
            else:
                os.unlink(tmp_file.name)
                return {'success': False, 'error': 'Could not read image'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def split_image_into_halves(image_path: str, filename: str) -> Dict:
    """Split image into left and right halves and save them"""
    try:
        print(f"🔧 Splitting image: {filename}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image for splitting'}
        
        height, width = image.shape[:2]
        mid_point = width // 2
        
        print(f"📏 Image dimensions: {width}x{height}, splitting at {mid_point}")
        
        # Split image into left and right halves
        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]
        
        # Save split images with unique names
        name, ext = os.path.splitext(filename)
        timestamp = str(int(time.time()))
        left_filename = f"{name}_left_{timestamp}{ext}"
        right_filename = f"{name}_right_{timestamp}{ext}"
        
        # Create upload folder path (will be set by caller)
        base_dir = os.path.dirname(image_path)
        left_path = os.path.join(base_dir, left_filename)
        right_path = os.path.join(base_dir, right_filename)
        
        # Save the split images
        cv2.imwrite(left_path, left_half)
        cv2.imwrite(right_path, right_half)
        
        print(f"✅ Split images saved:")
        print(f"   Left: {left_path}")
        print(f"   Right: {right_path}")
        
        return {
            'success': True,
            'left_path': left_path,
            'right_path': right_path,
            'left_filename': left_filename,
            'right_filename': right_filename,
            'original_dimensions': (width, height),
            'split_point': mid_point
        }
        
    except Exception as e:
        print(f"❌ Error splitting image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def convert_pdf_to_images(pdf_path: str, output_base_name: str, upload_folder: str, dpi: int = 200) -> List[Dict]:
    """Render a PDF into page images saved in upload folder.

    Returns a list of dicts with keys: page_index, image_filename, image_path
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not available for PDF processing")
        
    page_infos: List[Dict] = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        for i in range(total_pages):
            page = doc.load_page(i)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            page_filename = f"{output_base_name}_page_{i+1:03d}.png"
            page_path = os.path.join(upload_folder, page_filename)
            cv2.imwrite(page_path, img)
            page_infos.append({
                'page_index': i,
                'image_filename': page_filename,
                'image_path': page_path
            })
        doc.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise
    return page_infos

def generate_unique_filename(filename: str) -> str:
    """Generate unique filename with timestamp"""
    name, ext = os.path.splitext(filename)
    timestamp = str(int(time.time()))
    return f"{name}_{timestamp}{ext}"

def save_json_data(data: Dict, filepath: str) -> bool:
    """Save data to JSON file"""
    try:
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON data to {filepath}: {e}")
        return False

def load_json_data(filepath: str) -> Optional[Dict]:
    """Load data from JSON file"""
    try:
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading JSON data from {filepath}: {e}")
        return None

def create_inference_data(filename: str, original_text: str = "",
                         pre_llm_text: str = "",
                         corrected_text: str = "",
                         manual_text: str = "",
                         line_segments: List = None, total_lines: int = 0,
                         pipeline: str = "unknown", gemini_processing: bool = False,
                         llm_corrected: bool = False,
                         manually_edited: bool = False,
                         **kwargs) -> Dict:
    """Create standardized inference data structure"""
    return {
        'image': filename,
        'original_text': original_text,
        'pre_llm_text': pre_llm_text if pre_llm_text else original_text,
        'corrected_text': corrected_text,
        'manual_text': manual_text,
        'line_segments': line_segments or [],
        'total_lines': total_lines,
        'pipeline': pipeline,
        'gemini_processing': gemini_processing,
        'llm_corrected': llm_corrected,
        'manually_edited': manually_edited,
        'timestamp': time.time(),
        **kwargs
    }

def update_line_segment_filenames(line_segments: List[Dict], prefix: str, line_segments_folder: str) -> None:
    """Update line segment filenames with prefix and rename files"""
    try:
        for seg in line_segments:
            img_name = seg.get('image_filename')
            if not img_name:
                continue
            
            # Avoid double prefixing
            if not img_name.startswith(prefix):
                old_path = os.path.join(line_segments_folder, img_name)
                new_name = f"{prefix}_{img_name}"
                new_path = os.path.join(line_segments_folder, new_name)
                if os.path.exists(old_path):
                    try:
                        os.replace(old_path, new_path)
                    except Exception:
                        pass
                seg['image_filename'] = new_name
    except Exception as e:
        print(f"Warning: could not rename line segment images: {e}")

def adjust_line_indices_for_continuation(line_segments: List[Dict], start_index: int) -> None:
    """Adjust line indices for continuation from previous segments"""
    for segment in line_segments:
        segment['line_index'] += start_index
        segment['reading_order_index'] += start_index
        segment['position_in_column'] += start_index

def combine_text_from_segments(line_segments: List[Dict],
                                prefer: str = 'best') -> str:
    """Combine text from line segments using specified text field.

    prefer options:
        best     - manual > corrected > pre_llm > ocr_text
        manual   - ocr_text_manual only
        corrected - ocr_text_corrected only
        pre_llm  - ocr_text_pre_llm only
        raw      - ocr_text only
    """
    if not line_segments:
        return ""

    sorted_segments = sorted(line_segments,
                             key=lambda x: x.get('line_index', 0))
    text_lines = []

    for segment in sorted_segments:
        if prefer == 'manual':
            text = segment.get('ocr_text_manual', '')
        elif prefer == 'corrected':
            text = segment.get('ocr_text_corrected', '')
        elif prefer == 'pre_llm':
            text = segment.get('ocr_text_pre_llm', '')
        elif prefer == 'raw':
            text = segment.get('ocr_text', '')
        else:
            # best available
            text = (
                segment.get('ocr_text_manual') or
                segment.get('ocr_text_corrected') or
                segment.get('ocr_text_pre_llm') or
                segment.get('ocr_text', '')
            )

        if text and text.strip():
            text_lines.append(text.strip())

    return "\n".join(text_lines)

def get_last_two_lines(text: str) -> str:
    """Extract last two lines from text for context"""
    if not text:
        return ""
    
    lines = text.strip().split('\n')
    if len(lines) >= 2:
        return '\n'.join(lines[-2:])
    elif len(lines) == 1:
        return lines[0]
    return ""

def validate_file_size(file_size: int, max_size: int) -> bool:
    """Validate file size against maximum allowed size"""
    return file_size <= max_size

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    return secure_filename(filename)

def create_directory_structure(config) -> None:
    """Create necessary directory structure"""
    directories = [
        config.upload_folder,
        config.annotation_folder,
        config.inference_folder,
        config.line_segments_folder
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {file_path}: {e}")

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def is_pdf_file(filename: str) -> bool:
    """Check if file is a PDF"""
    return get_file_extension(filename) == 'pdf'

def is_image_file_extension(filename: str, image_extensions: set) -> bool:
    """Check if file has image extension"""
    return get_file_extension(filename) in image_extensions
