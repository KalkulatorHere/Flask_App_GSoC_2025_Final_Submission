import torch

print(f"PyTorch version:        {torch.__version__}")
print(f"CUDA available:         {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"GPU Name:               {torch.cuda.get_device_name(0)}")
print(f"GPU Count:              {torch.cuda.device_count()}")
print(f"cuDNN version:          {torch.backends.cudnn.version()}")

'''
HERE
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose 2>&1 | Tee-Object -FilePath llama_build.log


# Create the virtual environment
python -m venv renaissance_ocr

# Activate it
renaissance_ocr/Scripts/activate

# setup tools and wheel
python -m pip install --upgrade pip setuptools wheel

#pytorch install - check best cuda comaptibility option
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# HuggingFace stack (Transformers 5.8.0 is current as of May 5, 2026)
pip install transformers accelerate sentencepiece

# Flask web framework
pip install flask

#
pip install flask-cors

# Image processing
pip install Pillow opencv-python

# Scientific computing
pip install numpy matplotlib scipy scikit-image

# HuggingFace datasets & evaluation
pip install datasets evaluate

# OCR metrics (Character Error Rate / Word Error Rate)
pip install jiwer

# PDF and image pipeline
pip install pdf2image albumentations

# Progress bars
pip install tqdm

# Jupyter notebooks (for the .ipynb files in the repo)
pip install jupyter notebook ipykernel

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=span_ocr --display-name "Spanish OCR"
'''