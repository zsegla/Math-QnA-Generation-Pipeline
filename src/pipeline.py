#src/pipeline.py
import json
import os
import pandas as pd
from pathlib import Path
from typing import List

from src.config import settings
from src.generator.qna_generator import QnAGenerator
from src.utils.pdf_to_image_splitter import split_pdf_pages_as_image_pdfs
from src.utils import gemini_api_utils
from src.utils.gemini_api import model 

class MathQnAPipeline:
    def __init__(
        self,
        pdf_path: Path = settings.PDF_PATH,
        image_dir: Path = settings.IMAGE_DIR,
        output_dir: Path = settings.OUTPUT_DIR,
        temp_image_output_dir: Path = settings.TEMP_IMAGE_OUTPUT_DIR,
        max_pdf_pages_to_process: int = 7, # Default is 7
        process_images_flag: bool = True,
        process_pdfs_flag: bool = False
    ):
        self.pdf_path = pdf_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.temp_image_output_dir = temp_image_output_dir
        self.max_pdf_pages_to_process = max_pdf_pages_to_process
        self.process_images_flag = process_images_flag
        self.process_pdfs_flag = process_pdfs_flag

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_image_output_dir, exist_ok=True)

        # Initialize QnAGenerator with the model and output_dir
        self.qna_generator = QnAGenerator(model=model, output_dir=self.output_dir)
        
        # Configure Gemini API once at pipeline initialization
        gemini_api_utils.configure_gemini()

    def run(self):
        """
        Runs the complete Math Q&A Generation Pipeline using Gemini for image processing.
        """
        print("Starting Math Q&A Generation Pipeline...")

        all_source_image_paths: List[Path] = []
        
        # --- Phase 1: Collect Image Paths from Image Directory ---
        if self.process_images_flag:
            print("\n--- Phase 1: Collecting Image Paths from Image Directory ---")
            supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff'] 
            for ext in supported_extensions:
                all_source_image_paths.extend(self.image_dir.glob(ext))
            
            if not all_source_image_paths:
                print(f"No image files found in '{self.image_dir}'.")
            else:
                print(f"[✓] Found {len(all_source_image_paths)} image(s) in '{self.image_dir}'.")

        # --- Phase 1.5: Split PDF into Image-based Pages and Collect Paths ---
        if self.process_pdfs_flag and self.pdf_path and self.pdf_path.exists():
            print("\n--- Phase 1.5: Splitting PDF into Image-based Pages ---")
            
            # Pass the max_pages limit directly to the splitter function
            pdf_page_image_paths = split_pdf_pages_as_image_pdfs(
                input_pdf_path=self.pdf_path,
                output_dir=self.temp_image_output_dir,
                max_pages=self.max_pdf_pages_to_process # <--- THIS IS THE KEY CHANGE
            )

            all_source_image_paths.extend(pdf_page_image_paths)
            if pdf_page_image_paths:
                print(f"[✓] Converted {len(pdf_page_image_paths)} pages from PDF to images.")
            else:
                print("No pages converted from PDF.")
        elif self.process_pdfs_flag and (not self.pdf_path or not self.pdf_path.exists()):
            print(f"[!] PDF processing enabled, but PDF file not found at: {self.pdf_path}. Skipping PDF splitting.")

        if not all_source_image_paths:
            print("No source image files (from image directory or PDF) to process. Exiting pipeline.")
            return

        print(f"\nTotal unique image sources to process: {len(all_source_image_paths)}")

        extracted_qna_data_combined = []

        # --- Phase 2: Generating Q&A from All Collected Images ---
        print("\n--- Phase 2: Generating Q&A from Images using Gemini ---")
        for i, image_path in enumerate(all_source_image_paths):
            print(f"\n--- Processing Image {i+1}/{len(all_source_image_paths)}: {image_path.name} ---")
            
            qna_result = self.qna_generator.process_image_for_qna(image_path)
            
            if qna_result:
                # Determine source file based on the directory the image came from
                if self.image_dir in image_path.parents:
                    qna_result["source_file"] = image_path.name
                elif self.temp_image_output_dir in image_path.parents:
                    qna_result["source_file"] = self.pdf_path.name if self.pdf_path else "PDF_Source"
                else:
                    qna_result["source_file"] = "Unknown"

                qna_result["formula_id"] = f"{Path(qna_result['source_file']).stem}_{image_path.stem}_formula_{i+1}"
                qna_result["image"] = str(image_path.name)

                extracted_qna_data_combined.append(qna_result)
                print(f"  [✓] Successfully processed {image_path.name}.")
            else:
                print(f"  [-] No Q&A generated for {image_path.name}.")

        if not extracted_qna_data_combined:
            print("No math expressions or Q&A generated from any source images. Exiting pipeline.")
            return

        print(f"\n[✓] Finished processing all images. Generated Q&A for {len(extracted_qna_data_combined)} items.")

        # --- Phase 3: Saving Final Results ---
        print("\n--- Phase 3: Saving Final Results ---")
        final_qna_dataset_json_path = self.output_dir / "final_qna_dataset.json"
        final_qna_dataset_csv_path = self.output_dir / "final_qna_dataset.csv"

        with open(final_qna_dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_qna_data_combined, f, indent=4)
        print(f"[+] Saving {len(extracted_qna_data_combined)} Q&A results to {final_qna_dataset_json_path}...")
        print("[✓] Results saved successfully.")
        
        try:
            df = pd.DataFrame(extracted_qna_data_combined)
            for col in ['alternate_answers', 'critical_expressions', 'topics', 'critical_steps']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x))
            
            df.to_csv(final_qna_dataset_csv_path, index=False, encoding='utf-8-sig')
            print(f"[✓] Formatted results also saved directly to CSV: {final_qna_dataset_csv_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

        print("Pipeline execution complete.")

        # Overall Metrics for the entire run
        total_prompt_tokens = sum(r.get('total_prompt_tokens_per_question', 0) for r in extracted_qna_data_combined)
        total_candidate_tokens = sum(r.get('total_candidate_tokens_per_question', 0) for r in extracted_qna_data_combined)
        total_api_time = sum(r.get('total_api_time_per_question', 0.0) for r in extracted_qna_data_combined)

        print(f"\nOverall Metrics for this Run:")
        print(f"  Total Prompt Tokens Used: {total_prompt_tokens}")
        print(f"  Total Candidate Tokens Generated: {total_candidate_tokens}")
        print(f"  Total API Time Spent: {total_api_time:.2f} seconds")