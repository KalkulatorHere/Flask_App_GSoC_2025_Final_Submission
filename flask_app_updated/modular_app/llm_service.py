"""
LLM Service - Handles text correction using multiple backends with fallback system
Supports Gemini, HuggingFace, and Local LLaMA with automatic fallback
"""
import os
import time
import logging
import requests
from typing import Tuple, List, Dict, Optional

# Optional imports with guards
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("Gemini API integration available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini API not available - install google-generativeai")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.info("llama-cpp-python not available")

logger = logging.getLogger(__name__)

class LLMService:
    """Service for text correction using multiple LLM backends"""
    
    def __init__(self, config):
        self.config = config
        self.local_llama_client = None
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Setup Gemini API if available"""
        if GEMINI_AVAILABLE and self.config.gemini_api_key:
            try:
                genai.configure(api_key=self.config.gemini_api_key)
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.error(f"Error configuring Gemini API: {e}")
        else:
            logger.info("Gemini API not available - missing key or library")
    
    def chunk_text(self, text: str, chunk_size_chars: int = None) -> List[str]:
        """Split text into chunks if it exceeds threshold"""
        if chunk_size_chars is None:
            chunk_size_chars = self.config.fallback_chunk_size
        
        if len(text) <= chunk_size_chars:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size_chars):
            chunks.append(text[i:i + chunk_size_chars])
        
        return chunks
    
    def create_correction_prompt(self, text: str, context: str = "") -> str:
        """Create structured prompt for text correction with context"""
        context_section = ""
        if context:
            context_section = f"""
Previous context:
{context}

"""
        
        return f"""
Correct the following Spanish OCR text while preserving original grammar and style.
Only fix orthographic errors, punctuation, and obvious OCR mistakes.
{context_section}
Text to correct:
{text}

Instructions:
- Fix spelling errors and OCR artifacts
- Preserve historical language patterns  
- Maintain original formatting
- Return ONLY the corrected text

Corrected text:
"""
    
    def try_gemini_correction(self, text: str, context: str = "", retries: int = 2) -> Tuple[str, str]:
        """Try Gemini API for text correction"""
        if not GEMINI_AVAILABLE or not self.config.gemini_api_key:
            return text, "gemini_unavailable"
        
        prompt = self.create_correction_prompt(text, context)
        
        for attempt in range(retries):
            try:
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
                if response.candidates and response.text:
                    return response.text.strip(), "gemini"
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return text, "gemini_failed"
    
    def try_huggingface_correction(self, text: str, context: str = "", retries: int = 2) -> Tuple[str, str]:
        """Try Hugging Face Inference API for text correction"""
        if not self.config.hf_api_token:
            return text, "hf_unavailable"
        
        prompt = self.create_correction_prompt(text, context)
        
        # Use a general text generation model
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {self.config.hf_api_token}"}
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers,
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
                    timeout=self.config.llm_timeout_seconds
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                        if generated_text and generated_text != prompt:
                            return generated_text.replace(prompt, '').strip(), "hf"
                
            except Exception as e:
                logger.warning(f"HuggingFace attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return text, "hf_failed"
    
    def init_local_llama(self) -> bool:
        """Initialize local LLaMA client lazily"""
        if self.local_llama_client is not None:
            return True
        
        if not LLAMA_CPP_AVAILABLE or not self.config.llama_model_path:
            return False
        
        if not os.path.exists(self.config.llama_model_path):
            logger.warning(f"LLaMA model path not found: {self.config.llama_model_path}")
            return False
        
        try:
            # Ensure you accepted Llama 3.1 Community License before downloading weights. 
            # See: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
            self.local_llama_client = Llama(
                model_path=self.config.llama_model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            logger.info("Local LLaMA model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local LLaMA: {e}")
            return False
    
    def try_local_llama_correction(self, text: str, context: str = "") -> Tuple[str, str]:
        """Try local LLaMA for text correction"""
        if not self.init_local_llama():
            return text, "local_llama_unavailable"
        
        prompt = self.create_correction_prompt(text, context)
        
        try:
            response = self.local_llama_client(
                prompt,
                max_tokens=512,
                temperature=0.3,
                stop=["Human:", "\n\n"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            if generated_text:
                return generated_text, "local_llama"
                
        except Exception as e:
            logger.error(f"Local LLaMA correction failed: {e}")
        
        return text, "local_llama_failed"
    
    def process_text_with_fallbacks(self, text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> Tuple[str, str]:
        """
        Main fallback wrapper function that tries all LLM backends in order:
        1. Gemini (primary)
        2. HuggingFace (secondary)  
        3. Local LLaMA (final fallback)
        """
        
        # Try Gemini first
        logger.info("Attempting Gemini correction...")
        result, status = self.try_gemini_correction(text, context, retries)
        if status == "gemini":
            logger.info("Used: gemini")
            return result, status
        
        # Fallback to HuggingFace
        logger.info("Fallback: attempting HuggingFace correction...")
        result, status = self.try_huggingface_correction(text, context, retries)
        if status == "hf":
            logger.info("Fallback: hf") 
            return result, status
        
        # Final fallback to local LLaMA
        logger.info("Fallback: attempting local LLaMA correction...")
        result, status = self.try_local_llama_correction(text, context)
        if status == "local_llama":
            logger.info("Fallback: local_llama")
            return result, status
        
        # All backends failed
        logger.warning("All LLM backends failed, returning original text")
        return text, "all_failed"
    
    def process_text_with_chunking(self, text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> Tuple[str, str]:
        """Process text with chunking support for large inputs"""
        if len(text) <= self.config.fallback_chunk_size:
            return self.process_text_with_fallbacks(text, context, max_tokens, retries)
        
        # Chunk the text
        chunks = self.chunk_text(text, self.config.fallback_chunk_size)
        logger.info(f"Processing {len(chunks)} chunks...")
        
        corrected_chunks = []
        current_context = context
        final_status = "all_failed"
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            corrected_chunk, status = self.process_text_with_fallbacks(chunk, current_context, max_tokens, retries)
            corrected_chunks.append(corrected_chunk)
            
            # Update status to first successful one
            if final_status == "all_failed" and status != "all_failed":
                final_status = status
            
            # Update context for next chunk (last 2 lines)
            chunk_lines = corrected_chunk.strip().split('\n')
            if len(chunk_lines) >= 2:
                current_context = '\n'.join(chunk_lines[-2:])
            elif len(chunk_lines) == 1:
                current_context = chunk_lines[0]
        
        return '\n'.join(corrected_chunks), final_status
    
    def get_last_two_lines(self, text: str) -> str:
        """Extract last two lines from text for context"""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return '\n'.join(lines[-2:])
        elif len(lines) == 1:
            return lines[0]
        return ""
    
    def process_text_with_gemini(self, text: str, context: str = "", max_retries: int = 1) -> Tuple[str, str]:
        """Updated to use new three-tier fallback system"""
        result, status = self.process_text_with_fallbacks(text, context, retries=max_retries)
        
        # Map new status codes to legacy expected return values
        if status in ["gemini", "hf", "local_llama"]:
            return result, "success"
        else:
            return text, "max_retries_exceeded"
    
    def process_line_segments_with_gemini(self, line_segments: List[Dict], context: str = "") -> List[Dict]:
        """Process line segments with Gemini API for text correction"""
        if not GEMINI_AVAILABLE:
            return line_segments
        
        corrected_segments = []
        previous_context = context
        
        for i, segment in enumerate(line_segments):
            original_text = segment.get('ocr_text', '')
            if not original_text.strip():
                corrected_segments.append(segment)
                continue
            
            # Process with Gemini
            corrected_text, status = self.process_text_with_gemini(original_text, previous_context)
            
            # Update segment with corrected text
            corrected_segment = segment.copy()
            corrected_segment['ocr_text_corrected'] = corrected_text
            corrected_segment['correction_status'] = status
            corrected_segments.append(corrected_segment)
            
            # Update context for next line (use last 2 lines of corrected text)
            if status == "success":
                previous_context = self.get_last_two_lines(corrected_text)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return corrected_segments
    
    def manual_fallback_test(self) -> Tuple[str, str]:
        """Test function to verify fallback system"""
        test_text = "Prueba: texto con errores OCR para verificar el sistema de fallback..."
        logger.info("=== Manual Fallback Test ===")
        logger.info(f"Input: {test_text}")
        
        result, status = self.process_text_with_fallbacks(test_text)
        
        logger.info(f"Output: {result}")
        logger.info(f"Backend used: {status}")
        logger.info("=== Test Complete ===")
        
        return result, status
