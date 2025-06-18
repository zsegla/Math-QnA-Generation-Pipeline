Got it. Here's a comprehensive `README.md` file for your Math Q\&A Generation project, covering environment setup, API key configuration, and running the pipeline.

````markdown
# Math Q&A Generation Pipeline

This project leverages the Gemini API to process mathematical images (from stand-alone images or PDF documents) to extract LaTeX expressions, generate questions, solutions, and related metadata. It then evaluates the generated solutions and organizes the data into a structured dataset.

## Table of Contents

1.  [Features](#features)
2.  [Prerequisites](#prerequisites)
3.  [Setup Instructions](#setup-instructions)
    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
    * [3. Install System Dependencies (Poppler)](#3-install-system-dependencies-poppler)
    * [4. Install Python Dependencies](#4-install-python-dependencies)
    * [5. Obtain and Configure Gemini API Key](#5-obtain-and-configure-gemini-api-key)
4.  [Project Structure](#project-structure)
5.  [Usage](#usage)
6.  [Configuration](#configuration)
7.  [Output](#output)
8.  [Troubleshooting](#troubleshooting)
9.  [License](#license)

## 1. Features

* **Image Processing:** Extracts mathematical expressions (LaTeX) from image files.
* **Q&A Generation:** Generates questions, detailed solutions, and final answers from mathematical content using Gemini (LLM1).
* **Solution Validation:** Generates a proposed solution using a second LLM (LLM2) and evaluates it against the first LLM's solution.
* **Metadata Extraction:** Identifies critical expressions, topics, and critical steps.
* **Alternate Answers:** Generates alternative correct answers.
* **PDF Integration:** Can process pages from a PDF document by converting them into images.
* **Structured Output:** Saves generated Q&A data into both JSON and CSV formats.
* **Token & Time Tracking:** Tracks API token usage and time spent per question and overall.

## 2. Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python 3.8+**: This project is developed with Python 3.
* **Homebrew (macOS)**: Recommended for easily installing Poppler. If you are on Linux or Windows, you will need to find the equivalent way to install Poppler.

## 3. Setup Instructions

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd project-01-b-main       # Navigate into the project directory
````

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with your system's Python packages.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

You will know the virtual environment is active when your terminal prompt changes to include `(venv)` at the beginning, like `(venv) yourusername@yourcomputer project-01-b-main %`.

### 3\. Install System Dependencies (Poppler)

The `pdf2image` Python library relies on an external utility called **Poppler**. You need to install Poppler on your operating system.

**For macOS (using Homebrew):**

If you don't have Homebrew, install it first:

```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
# Follow any on-screen instructions to complete Homebrew installation.
```

Then, install Poppler:

```bash
brew install poppler
```

**For Linux (e.g., Debian/Ubuntu):**

```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**For Windows:**
Installing Poppler on Windows usually involves downloading pre-compiled binaries and adding them to your system's PATH. Refer to the `pdf2image` documentation or search for "install poppler windows" for detailed instructions. A common approach is to use conda or download from sites like poppler.freedesktop.org.

### 4\. Install Python Dependencies

With your virtual environment active, install all the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This command will install `pandas`, `google-generativeai`, `python-dotenv`, `Pillow`, `pdf2image`, and `PyPDF2`.

### 5\. Obtain and Configure Gemini API Key

This project interacts with the Google Gemini API, which requires an API key for authentication.

1.  **Obtain your API Key:**

      * Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or the Google Cloud Console.
      * Follow the instructions to create a new API key for the Gemini API. Your key will look something like `AIzaSyB-C1D2E3F4G5H6I7J8K9L0M1N2O3P4Q5R6`.

2.  **Create a `.env` file:**

      * In the **root directory of your project** (the same directory where `main.py` is located), create a new file named `.env`.

      * Open this `.env` file in a plain text editor and add the following line, replacing `YOUR_ACTUAL_GEMINI_API_KEY` with the key you obtained in the previous step:

        ```
        API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
        ```

      * **Save** the `.env` file.

    *Example `.env` content:*

    ```
    API_KEY=AIzaSyB-xxxxxxxxxxxxxxxxxxxxxxxxx
    ```

    **Important:** Do not commit your `.env` file to version control (e.g., Git), as it contains sensitive credentials. It's already included in `.gitignore` to prevent accidental commits.

## 4\. Project Structure

```
.
├── data/
│   ├── images/                 # Put your input image files (.png, .jpg, etc.) here
│   └── raw_pdfs/               # Put your input PDF files here
├── outputs/                    # Output directory for generated Q&A data and temporary images
│   ├── temp_page_images/       # Temporary directory for images split from PDFs
│   └── final_qna_dataset.csv   # Final Q&A data in CSV format
│   └── final_qna_dataset.json  # Final Q&A data in JSON format
├── src/
│   ├── generator/
│   │   └── qna_generator.py    # Core logic for Q&A generation from images
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gemini_api.py       # Handles Gemini API configuration and model instantiation
│   │   ├── gemini_api_utils.py # Utility functions for specific Gemini API calls (e.g., extract latex, generate Q&A)
│   │   └── pdf_to_image_splitter.py # Utility for converting PDF pages to images
│   ├── __init__.py
│   ├── config.py               # Project settings, paths, and API key reference
│   └── pipeline.py             # Orchestrates the entire Q&A generation process
├── main.py                     # Entry point to run the pipeline
└── requirements.txt            # Lists all Python dependencies
```

## 5\. Usage

Once all prerequisites are installed and your API key is configured, you can run the Q\&A generation pipeline:

1.  **Place your input files:**

      * Place any image files (`.png`, `.jpg`, etc.) you want to process in the `data/images/` directory.
      * Place any PDF files you want to process in the `data/raw_pdfs/` directory. (The default PDF is `Abramowitz & Stegun.pdf` in `src/config.py`).

2.  **Configure `main.py` (Optional):**
    Open `main.py` to adjust flags for processing images vs. PDFs and limit the number of PDF pages processed.

    ```python
    # main.py
    from src.pipeline import MathQnAPipeline
    from src.config import settings

    def main():
        pipeline = MathQnAPipeline(
            pdf_path=settings.PDF_PATH,
            image_dir=settings.IMAGE_DIR,
            output_dir=settings.OUTPUT_DIR,
            max_pdf_pages_to_process=7,  # Process first 7 pages of the PDF
            process_images_flag=True,    # Set to True to process images from `data/images`
            process_pdfs_flag=False      # Set to True to process the PDF from `data/raw_pdfs`
        )
        pipeline.run()

    if __name__ == "__main__":
        main()
    ```

      * Set `process_images_flag=True` to process images found in `data/images/`.
      * Set `process_pdfs_flag=True` and provide `pdf_path` to process a PDF.
      * `max_pdf_pages_to_process` controls how many pages from the PDF are converted to images and processed.

3.  **Run the Pipeline:**
    Execute the `main.py` script from your project's root directory:

    ```bash
    python3 main.py
    ```

The script will print progress messages to the console as it processes images and generates Q\&A data.

## 6\. Configuration

  * **`src/config.py`**:
    This file holds important path configurations and references your Gemini API key. You can modify `PDF_PATH` to point to a different default PDF.

## 7\. Output

Upon successful execution, the generated Q\&A dataset will be saved in the `outputs/` directory:

  * `outputs/final_qna_dataset.json`: A JSON file containing all the extracted and generated Q\&A data.
  * `outputs/final_qna_dataset.csv`: A CSV file with the same data, suitable for spreadsheet analysis.

Temporary image files generated from PDFs will be stored in `outputs/temp_page_images/`.

## 8\. Troubleshooting

  * **`ModuleNotFoundError`**: If you encounter this error, it means a required Python package is missing. Ensure your virtual environment is active and run `pip install -r requirements.txt`. If the error persists, manually install the missing package (e.g., `pip install <package-name>`).
  * **`KeyError: 'API_KEY'`**: This means your `API_KEY` is not correctly set in the `.env` file in the project's root directory. Double-check the file name (`.env`), its location, and the `API_KEY=YOUR_KEY_HERE` format.
  * **Network/Connection Errors (`NewConnectionError`) during `pip install`**: This indicates a problem connecting to PyPI. Check your internet connection, DNS settings, VPN/proxy, and firewall.
  * **`pdf2image` errors (even after `pip install pdf2image`)**: This most likely means **Poppler** is not correctly installed or not found in your system's PATH. Re-verify Poppler installation using Homebrew (macOS) or your OS's package manager.
  * **Virtual Environment not activating**: Ensure you are in the correct project directory and using `source venv/bin/activate`. The prompt should change.

## 9\. Author

Fekreselassie Solomon Mulu