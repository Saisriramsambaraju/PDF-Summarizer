# PDFSummarizer

PDFSummarizer is a Python-based tool designed to extract text from PDF files, summarize the extracted text, and convert text to speech. It leverages state-of-the-art machine learning models and OCR technologies to provide a seamless experience for processing PDF documents.

## Features

- **PDF Text Extraction**: Convert PDF pages to images and use OCR to extract text from each page.
- **Text Summarization**: Summarize extracted text using the BART (Bidirectional and Auto-Regressive Transformers) model.
- **Text-to-Speech (TTS) Synthesis**: Convert text to speech using the `melo` TTS library with support for multiple languages and accents.
- **Command-Line Interface (CLI)**: Easily specify file paths, modes, and output options through a user-friendly CLI.

## Requirements

- Python 3.6+
- `argparse`
- `pdf2image`
- `pytesseract`
- `torch`
- `transformers`
- `melo`
- `datasets`
- Tesseract OCR

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/PDFSummarizer.git
    cd PDFSummarizer
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```
3. Install Melo-api:
     ```sh
     git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download
     ```
4. Install Tesseract OCR:

    - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
    - **macOS**: Install via Homebrew:

        ```sh
        brew install tesseract
        ```

    - **Linux**: Install via package manager:

        ```sh
        sudo apt-get install tesseract-ocr
        ```

## Usage

Run the script with the following command:

```sh
python pdfsummarizer.py <path> <mode> [--output <output>]
