import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

POPPLER_URL = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.02.0-0/Release-24.02.0-0.zip"

BASE_DIR = Path(__file__).parent
DEPS_DIR = BASE_DIR / "deps"

def install_poppler():
    poppler_dir = DEPS_DIR / "poppler"
    if poppler_dir.exists():
        print("Poppler already installed in deps/poppler")
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
            os.rename(extracted_dirs[0], poppler_dir)
            print(f"Poppler installed to {poppler_dir}")
        else:
            print("Could not find extracted Poppler folder")
            
    except Exception as e:
        print(f"Failed to install Poppler: {e}")
    finally:
        if zip_path.exists():
            os.remove(zip_path)

if __name__ == "__main__":
    install_poppler()
