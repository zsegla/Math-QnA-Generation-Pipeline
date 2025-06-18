# main.py
from src.pipeline import MathQnAPipeline
from src.config import settings 


def main():
    """
    Main entry point for the Math Q&A Generation Pipeline.
    Instantiates and runs the pipeline.
    """
    pipeline = MathQnAPipeline(
        pdf_path=settings.PDF_PATH,
        image_dir=settings.IMAGE_DIR,
        output_dir=settings.OUTPUT_DIR,
        max_pdf_pages_to_process=7, # Example: process first 7 pages of PDF
        process_images_flag=True,   # Set to True to enable image processing
        process_pdfs_flag=False      # Set to True to enable PDF processing
    )
    pipeline.run()

if __name__ == "__main__":
    main()