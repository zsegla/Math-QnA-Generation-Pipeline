#src/utils/pdf_to_image_splitter.py
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path 
from PyPDF2 import PdfReader 

def split_pdf_pages_as_image_pdfs(input_pdf_path: Path, output_dir: Path, max_pages: Optional[int] = None) -> List[Path]:
    """
    Splits a multi-page PDF into individual IMAGE files (e.g., PNG),
    where each output file is the image representation of the corresponding page.
    Optionally limits the number of pages processed (starting from page 1).
    Returns a list of paths to the generated image files.
    """
    output_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

    print(f"[*] Processing PDF: {input_pdf_path.name}")
    print(f"[*] Saving individual page images (as PNGs) to: {output_dir}") # Updated message

    generated_image_paths = []
    
    total_pdf_pages = 0
    try:
        reader = PdfReader(str(input_pdf_path))
        total_pdf_pages = len(reader.pages)
        print(f"[*] Total pages in PDF: {total_pdf_pages}")
    except Exception as e:
        print(f"[X] Could not read PDF page count with PyPDF2: {e}. Proceeding assuming full PDF for max_pages calculation.")


    first_page_to_process = 1
    last_page_to_process = max_pages or None # Default to process all pages if no limit

    if max_pages is not None and max_pages > 0:
        last_page_to_process = min(total_pdf_pages, max_pages) if total_pdf_pages > 0 else max_pages
        print(f"[*] Limiting PDF processing to pages {first_page_to_process} to {last_page_to_process}.")

    try:
        images = convert_from_path(
            str(input_pdf_path),
            first_page=first_page_to_process,
            last_page=last_page_to_process
        )

        for i, image in enumerate(images):
            original_page_num = i + first_page_to_process
            # <--- KEY CHANGE HERE: Save as PNG instead of PDF
            output_filename = output_dir / f"{input_pdf_path.stem}_page_{original_page_num}.png"
            image.save(output_filename, "PNG") # <--- And specify PNG format
            generated_image_paths.append(output_filename)
            print(f"  [✓] Saved {output_filename.name}")
        
        print(f"[✓] Successfully split {len(images)} pages from {input_pdf_path.name}.")
        return generated_image_paths

    except Exception as e:
        print(f"[X] An error occurred during PDF splitting: {e}")
        print("[!] Ensure Poppler is installed and its 'bin' directory is in your system's PATH,")
        print("    or specify 'poppler_path' argument in convert_from_path().")
        return []