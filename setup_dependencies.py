import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import ctypes

POPPLER_URL = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.02.0-0/Release-24.02.0-0.zip"
TESSERACT_URL = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"

BASE_DIR = Path(__file__).parent
DEPS_DIR = BASE_DIR / "deps"
PROJECT_ROOT = BASE_DIR.parent

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_command(command, cwd=None, shell=True):
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        subprocess.check_call(command, cwd=cwd, shell=shell)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def install_python_deps():
    print("\n--- Installing Python Dependencies ---")
    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        run_command([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        print("requirements.txt not found in backend/")

def install_node_deps():
    print("\n--- Installing Node.js Dependencies ---")
    if (PROJECT_ROOT / "package.json").exists():
        run_command(["npm", "install"], cwd=PROJECT_ROOT)
    else:
        print("package.json not found in project root")

def install_poppler():
    print("\n--- Checking Poppler ---")
    poppler_dir = DEPS_DIR / "poppler"
    if poppler_dir.exists():
        print("Poppler already installed.")
        return

    print("Downloading Poppler...")
    DEPS_DIR.mkdir(exist_ok=True)
    zip_path = DEPS_DIR / "poppler.zip"
    
    try:
        urllib.request.urlretrieve(POPPLER_URL, zip_path)
        print("Extracting Poppler...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DEPS_DIR)
        
        extracted_dirs = [d for d in DEPS_DIR.iterdir() if d.is_dir() and "Release" in d.name]
        if extracted_dirs:
            if poppler_dir.exists():
                shutil.rmtree(poppler_dir)
            os.rename(extracted_dirs[0], poppler_dir)
            print(f"Poppler installed to {poppler_dir}")
        else:
            print("Could not find extracted Poppler folder")
            
    except Exception as e:
        print(f"Failed to install Poppler: {e}")
    finally:
        if zip_path.exists():
            os.remove(zip_path)

def install_tesseract():
    print("\n--- Checking Tesseract OCR ---")
    
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.environ.get("TESSERACT_PATH", "")
    ]
    
    found = False
    for path in common_paths:
        if path and os.path.exists(path):
            print(f"Tesseract found at: {path}")
            found = True
            break
            
    if found:
        return

    print("Tesseract not found. Installing...")
    DEPS_DIR.mkdir(exist_ok=True)
    installer_path = DEPS_DIR / "tesseract_setup.exe"
    
    try:
        print(f"Downloading installer from {TESSERACT_URL}...")
        urllib.request.urlretrieve(TESSERACT_URL, installer_path)
        
        print("Running installer... (Please accept the installation prompt)")
        print("Installing silently to default location...")
        
        cmd = [str(installer_path), "/S"]
        
        if not is_admin():
            print("Requesting admin privileges for installation...")
            ctypes.windll.shell32.ShellExecuteW(None, "runas", str(installer_path), "/S", None, 1)
        else:
            run_command(cmd, shell=False)
            
        print("Tesseract installation triggered. Please wait for it to complete.")
        print("NOTE: You may need to restart your terminal/IDE after installation for PATH updates to take effect.")
        
    except Exception as e:
        print(f"Failed to download/install Tesseract: {e}")
        print("Please install Tesseract manually from: https://github.com/UB-Mannheim/tesseract/wiki")

def main():
    print("=== RAG App Dependency Setup ===")
    install_python_deps()
    install_node_deps()
    install_poppler()
    install_tesseract()
    print("\n=== Setup Complete ===")
    print("If Tesseract was just installed, please restart your terminal.")

if __name__ == "__main__":
    main()
