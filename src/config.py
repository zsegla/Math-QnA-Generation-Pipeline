# src/config.py
from pathlib import Path
from src.utils.gemini_api import api_key
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings:
    BASE_DIR: Path = Path(__file__).parent.parent 
    
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw_pdfs"
    IMAGE_DIR: Path = DATA_DIR / "images" 
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    TEMP_IMAGE_OUTPUT_DIR: Path = OUTPUT_DIR / "temp_page_images" 

    # Default PDF path to process
    PDF_PATH: Path = RAW_PDF_DIR / "Abramowitz & Stegun.pdf"
    GEMINI_API_KEY = api_key or None
    
   
settings = Settings()