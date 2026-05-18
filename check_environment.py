"""
Environment snapshot script.
Run this to capture all package versions, CUDA, and driver info
needed to recreate this OCR install.
"""
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except Exception as e:
        return f"(not available: {e})"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Python ────────────────────────────────────────────────────
section("Python")
print(f"Executable : {sys.executable}")
print(f"Version    : {sys.version}")


# ── PyTorch + CUDA ────────────────────────────────────────────
section("PyTorch / CUDA")
try:
    import torch
    print(f"torch               : {torch.__version__}")
    print(f"CUDA available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version        : {torch.version.cuda}")
        print(f"cuDNN version       : {torch.backends.cudnn.version()}")
        print(f"GPU count           : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}             : {props.name}")
            print(f"  VRAM              : {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute cap.      : {props.major}.{props.minor}")
except ImportError:
    print("torch: NOT INSTALLED")


# ── NVIDIA Driver ─────────────────────────────────────────────
section("NVIDIA Driver (nvidia-smi)")
print(run("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader"))
print()
print(run("nvidia-smi"))


# ── Core ML / OCR packages ────────────────────────────────────
section("Core ML / OCR Packages")
packages = [
    "transformers",
    "tokenizers",
    "accelerate",
    "datasets",
    "huggingface-hub",
    "timm",
    "sentencepiece",
    "pillow",
    "opencv-python",
    "opencv-python-headless",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
]

for pkg in packages:
    try:
        import importlib.metadata
        version = importlib.metadata.version(pkg)
        print(f"{pkg:<35} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg:<35} NOT INSTALLED")


# ── Detectron2 ────────────────────────────────────────────────
section("Detectron2")
try:
    import detectron2
    print(f"detectron2          : {detectron2.__version__}")
except ImportError:
    print("detectron2          : NOT INSTALLED")

# Also try to get it from pip
d2_pip = run("pip show detectron2")
if "not available" not in d2_pip:
    for line in d2_pip.splitlines():
        if line.startswith(("Name:", "Version:", "Location:")):
            print(line)


# ── Flask / Web ───────────────────────────────────────────────
section("Flask / Web Packages")
web_packages = [
    "flask",
    "werkzeug",
    "flask-cors",
    "requests",
]
for pkg in web_packages:
    try:
        import importlib.metadata
        version = importlib.metadata.version(pkg)
        print(f"{pkg:<35} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg:<35} NOT INSTALLED")


# ── Optional LLM packages ─────────────────────────────────────
section("Optional LLM / API Packages")
optional = [
    "google-generativeai",
    "openai",
    "anthropic",
    "llama-cpp-python",
]
for pkg in optional:
    try:
        import importlib.metadata
        version = importlib.metadata.version(pkg)
        print(f"{pkg:<35} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg:<35} NOT INSTALLED")


# ── Full pip freeze ───────────────────────────────────────────
section("Full pip freeze (for requirements.txt)")
print(run("pip freeze"))


# ── Conda (if applicable) ─────────────────────────────────────
section("Conda Environment (if applicable)")
print(run("conda info"))
print()
print(run("conda list"))


print(f"\n{'='*60}")
print("  Done — copy this output to recreate your environment.")
print(f"{'='*60}\n")
