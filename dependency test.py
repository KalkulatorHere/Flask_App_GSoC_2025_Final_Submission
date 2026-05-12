import ctypes
import os

lib_path = r'C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR\span_ocr\Lib\site-packages\llama_cpp\lib\llama.dll'

# Try loading each DLL in the lib folder first
lib_dir = r'C:\Users\rdb104\Documents\caserepos\flask_app_TrOCR\span_ocr\Lib\site-packages\llama_cpp\lib'
for dll in os.listdir(lib_dir):
    if dll.endswith('.dll'):
        try:
            ctypes.CDLL(os.path.join(lib_dir, dll))
            print(f'OK:     {dll}')
        except Exception as e:
            print(f'FAILED: {dll} -> {e}')