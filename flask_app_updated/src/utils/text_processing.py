"""
Text processing utilities for Flask OCR Application
"""

import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    @staticmethod
    def extract_lines(text: str) -> List[str]:
        """Extract lines from text, filtering empty ones"""
        if not text:
            return []
        
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]
    
    @staticmethod
    def get_last_n_lines(text: str, n: int = 2) -> str:
        """Extract last N lines from text for context"""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        if len(lines) >= n:
            return '\n'.join(lines[-n:])
        elif len(lines) == 1:
            return lines[0]
        return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into chunks with optional overlap"""
        if not text or chunk_size <= 0:
            return [text] if text else []
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for last space within chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if overlap > 0 else end
        
        return chunks
    
    @staticmethod
    def merge_text_chunks(chunks: List[str], separator: str = '\n') -> str:
        """Merge text chunks with separator"""
        if not chunks:
            return ""
        
        return separator.join(chunk for chunk in chunks if chunk.strip())
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        if not text:
            return 0
        
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    @staticmethod
    def count_characters(text: str) -> int:
        """Count characters in text"""
        return len(text) if text else 0
    
    @staticmethod
    def count_lines(text: str) -> int:
        """Count lines in text"""
        if not text:
            return 0
        
        return len(text.split('\n'))
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def remove_duplicate_lines(text: str) -> str:
        """Remove duplicate lines from text"""
        if not text:
            return ""
        
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, any]:
        """Extract metadata from text"""
        if not text:
            return {
                'word_count': 0,
                'character_count': 0,
                'line_count': 0,
                'sentence_count': 0,
                'avg_words_per_line': 0,
                'avg_chars_per_line': 0
            }
        
        lines = TextProcessor.extract_lines(text)
        sentences = TextProcessor.extract_sentences(text)
        word_count = TextProcessor.count_words(text)
        character_count = TextProcessor.count_characters(text)
        line_count = len(lines)
        sentence_count = len(sentences)
        
        avg_words_per_line = word_count / line_count if line_count > 0 else 0
        avg_chars_per_line = character_count / line_count if line_count > 0 else 0
        
        return {
            'word_count': word_count,
            'character_count': character_count,
            'line_count': line_count,
            'sentence_count': sentence_count,
            'avg_words_per_line': round(avg_words_per_line, 2),
            'avg_chars_per_line': round(avg_chars_per_line, 2)
        }
    
    @staticmethod
    def validate_text_quality(text: str, min_length: int = 10) -> Dict[str, any]:
        """Validate text quality and return quality metrics"""
        if not text:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': ['Empty text']
            }
        
        issues = []
        quality_score = 1.0
        
        # Check minimum length
        if len(text.strip()) < min_length:
            issues.append(f'Text too short (less than {min_length} characters)')
            quality_score -= 0.3
        
        # Check for excessive whitespace
        if re.search(r'\s{5,}', text):
            issues.append('Excessive whitespace detected')
            quality_score -= 0.1
        
        # Check for repeated characters
        if re.search(r'(.)\1{4,}', text):
            issues.append('Repeated characters detected')
            quality_score -= 0.1
        
        # Check for non-printable characters
        if re.search(r'[^\x20-\x7E\n\r\t]', text):
            issues.append('Non-printable characters detected')
            quality_score -= 0.1
        
        # Check for very short lines (potential OCR errors)
        lines = text.split('\n')
        short_lines = [line for line in lines if len(line.strip()) < 3 and line.strip()]
        if len(short_lines) > len(lines) * 0.5:
            issues.append('Many very short lines detected')
            quality_score -= 0.2
        
        quality_score = max(0.0, quality_score)
        
        return {
            'is_valid': len(issues) == 0,
            'quality_score': quality_score,
            'issues': issues
        }
