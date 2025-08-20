# text_processor.py

import os
import re
import io

# Optional imports with fallbacks
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("⚠️ python-docx not available - DOCX files will be skipped")
    DOCX_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
    print("✅ Using PyMuPDF for PDF processing")
except ImportError:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
        from PyPDF2 import PdfReader
        PDF_AVAILABLE = True
        print("✅ Using pdfminer/PyPDF2 for PDF processing")
    except ImportError:
        print("⚠️ PDF libraries not available - PDF files will be skipped")
        PDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    print("⚠️ OCR libraries not available - OCR fallback disabled")
    OCR_AVAILABLE = False

def extract_text_from_file(file_path: str) -> str:
    """
    Dispatch to the correct reader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_word_file(file_path)
    elif ext in ['.txt', '.text']:
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_pdf_file(file_path: str) -> str:
    """
    Robust PDF text extraction with graceful fallbacks
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not PDF_AVAILABLE:
        print("⚠️ PDF processing not available")
        return ""

    text = ""
    
    # Try PyMuPDF first (most reliable)
    try:
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n".join(pages)
        print(f"[PyMuPDF] Extracted {len(text)} chars from {os.path.basename(file_path)}")
    except (NameError, Exception) as e:
        print(f"[PyMuPDF ERROR] {e}")
        
        # Fallback to pdfminer
        try:
            text = pdf_extract_text(file_path)
            print(f"[PDFMINER] Extracted {len(text)} chars from {os.path.basename(file_path)}")
        except (NameError, Exception) as e:
            print(f"[PDFMINER ERROR] {e}")
            
            # Fallback to PyPDF2
            try:
                reader = PdfReader(file_path)
                pages = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                text = "\n".join(pages)
                print(f"[PyPDF2] Extracted {len(text)} chars from {os.path.basename(file_path)}")
            except (NameError, Exception) as e:
                print(f"[PyPDF2 ERROR] {e}")

    # OCR fallback if still no text and OCR is available
    if not text.strip() and OCR_AVAILABLE:
        try:
            images = convert_from_path(file_path)
            ocr_pages = []
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                ocr_pages.append(ocr_text)
            text = "\n".join(ocr_pages)
            print(f"[OCR] Extracted {len(text)} chars via Tesseract from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[OCR ERROR] {e}")

    # Normalize whitespace and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    print(f"[PDF NORMALIZED] {len(normalized)} chars after cleanup")
    return normalized

def read_word_file(file_path: str) -> str:
    """
    Read text from a Word (.docx) file using python-docx with graceful fallback.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not DOCX_AVAILABLE:
        print("⚠️ DOCX processing not available")
        return ""

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        result = "\n".join(paragraphs)
        print(f"[DOCX] Extracted {len(result)} chars from {os.path.basename(file_path)}")
        return result
    except Exception as e:
        print(f"[DOCX ERROR] {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """
    Read plain text from a .txt file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[TXT] Extracted {len(content)} chars from {os.path.basename(file_path)}")
        return content
    except Exception as e:
        print(f"[TXT ERROR] {e}")
        return ""

def clean_up_text(text: str) -> str:
    """
    Clean raw text: collapse whitespace and remove unwanted characters.
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
    return text.strip()

def get_text_info(text: str) -> dict:
    """
    Return character, word, and sentence statistics from cleaned text.
    """
    if not text:
        return {
            'total_characters': 0,
            'total_words': 0,
            'total_sentences': 0,
            'average_words_per_sentence': 0
        }
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    total_sentences = len(sentences)
    return {
        'total_characters': len(text),
        'total_words': len(words),
        'total_sentences': total_sentences,
        'average_words_per_sentence': round(len(words) / max(total_sentences, 1), 2)
    }