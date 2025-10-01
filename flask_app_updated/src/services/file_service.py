"""
File management service for uploads, inference storage, and file operations
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional
from werkzeug.utils import secure_filename

from ..exceptions import FileProcessingException, ValidationException
from ..config import AppConfig

logger = logging.getLogger(__name__)


class FileService:
    """Service for file management operations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.config.files.allowed_extensions
    
    def is_image_file(self, filename: str) -> bool:
        """Check if file is an image"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.config.files.image_extensions
    
    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size"""
        return file_size <= self.config.files.max_file_size
    
    def generate_unique_filename(self, filename: str) -> str:
        """Generate unique filename with timestamp"""
        name, ext = os.path.splitext(filename)
        timestamp = str(int(time.time()))
        return f"{name}_{timestamp}{ext}"
    
    def save_uploaded_file(self, file, filename: str) -> str:
        """Save uploaded file to upload folder"""
        try:
            # Generate unique filename
            unique_filename = self.generate_unique_filename(filename)
            file_path = os.path.join(self.config.files.upload_folder, unique_filename)
            
            # Save file
            file.save(file_path)
            logger.info(f"File saved to: {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise FileProcessingException(f"Failed to save file: {e}")
    
    def save_inference_data(self, filename: str, inference_data: Dict) -> str:
        """Save inference data to JSON file"""
        try:
            inference_path = os.path.join(self.config.files.inference_folder, f"{filename}.json")
            
            # Add timestamp if not present
            if 'timestamp' not in inference_data:
                inference_data['timestamp'] = time.time()
            
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Inference data saved to: {inference_path}")
            return inference_path
            
        except Exception as e:
            logger.error(f"Error saving inference data: {e}")
            raise FileProcessingException(f"Failed to save inference data: {e}")
    
    def load_inference_data(self, filename: str) -> Optional[Dict]:
        """Load inference data from JSON file"""
        try:
            inference_path = os.path.join(self.config.files.inference_folder, f"{filename}.json")
            
            if not os.path.exists(inference_path):
                return None
            
            with open(inference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading inference data: {e}")
            raise FileProcessingException(f"Failed to load inference data: {e}")
    
    def update_inference_data(self, filename: str, updates: Dict) -> bool:
        """Update inference data with new information"""
        try:
            inference_data = self.load_inference_data(filename)
            if not inference_data:
                return False
            
            # Update with new data
            inference_data.update(updates)
            inference_data['last_updated'] = time.time()
            
            # Save updated data
            self.save_inference_data(filename, inference_data)
            return True
            
        except Exception as e:
            logger.error(f"Error updating inference data: {e}")
            return False
    
    def save_annotation_data(self, filename: str, annotation_data: Dict) -> str:
        """Save annotation data to JSON file"""
        try:
            annotation_path = os.path.join(self.config.files.annotation_folder, f"{filename}.json")
            
            # Add timestamp
            annotation_data['saved_at'] = time.time()
            
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Annotation data saved to: {annotation_path}")
            return annotation_path
            
        except Exception as e:
            logger.error(f"Error saving annotation data: {e}")
            raise FileProcessingException(f"Failed to save annotation data: {e}")
    
    def get_uploaded_images(self) -> List[str]:
        """Get list of uploaded image files"""
        try:
            if not os.path.exists(self.config.files.upload_folder):
                return []
            
            files = os.listdir(self.config.files.upload_folder)
            image_files = [f for f in files if self.is_image_file(f)]
            return sorted(image_files)
            
        except Exception as e:
            logger.error(f"Error getting uploaded images: {e}")
            return []
    
    def get_inference_files(self) -> List[str]:
        """Get list of inference files"""
        try:
            if not os.path.exists(self.config.files.inference_folder):
                return []
            
            files = os.listdir(self.config.files.inference_folder)
            json_files = [f for f in files if f.endswith('.json')]
            return sorted(json_files)
            
        except Exception as e:
            logger.error(f"Error getting inference files: {e}")
            return []
    
    def delete_file(self, filename: str, folder: str = 'uploads') -> bool:
        """Delete a file from specified folder"""
        try:
            if folder == 'uploads':
                file_path = os.path.join(self.config.files.upload_folder, filename)
            elif folder == 'inferences':
                file_path = os.path.join(self.config.files.inference_folder, filename)
            elif folder == 'annotations':
                file_path = os.path.join(self.config.files.annotation_folder, filename)
            elif folder == 'line_segments':
                file_path = os.path.join(self.config.files.line_segments_folder, filename)
            else:
                raise ValueError(f"Invalid folder: {folder}")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified hours"""
        try:
            cleaned_count = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Clean up upload folder
            for folder in [self.config.files.upload_folder, 
                          self.config.files.inference_folder,
                          self.config.files.annotation_folder,
                          self.config.files.line_segments_folder]:
                if not os.path.exists(folder):
                    continue
                
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old file: {file_path}")
            
            logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_file_info(self, filename: str, folder: str = 'uploads') -> Optional[Dict]:
        """Get file information"""
        try:
            if folder == 'uploads':
                file_path = os.path.join(self.config.files.upload_folder, filename)
            elif folder == 'inferences':
                file_path = os.path.join(self.config.files.inference_folder, filename)
            elif folder == 'annotations':
                file_path = os.path.join(self.config.files.annotation_folder, filename)
            elif folder == 'line_segments':
                file_path = os.path.join(self.config.files.line_segments_folder, filename)
            else:
                raise ValueError(f"Invalid folder: {folder}")
            
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            return {
                'filename': filename,
                'path': file_path,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_file': os.path.isfile(file_path),
                'is_directory': os.path.isdir(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check on file service"""
        return {
            'service_ready': True,
            'upload_folder_exists': os.path.exists(self.config.files.upload_folder),
            'inference_folder_exists': os.path.exists(self.config.files.inference_folder),
            'annotation_folder_exists': os.path.exists(self.config.files.annotation_folder),
            'line_segments_folder_exists': os.path.exists(self.config.files.line_segments_folder),
            'upload_folder_writable': os.access(self.config.files.upload_folder, os.W_OK) if os.path.exists(self.config.files.upload_folder) else False,
            'inference_folder_writable': os.access(self.config.files.inference_folder, os.W_OK) if os.path.exists(self.config.files.inference_folder) else False
        }
