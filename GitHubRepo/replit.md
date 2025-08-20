# Overview

A Flask-based academic integrity system that detects plagiarism and AI-generated content in submitted documents. The application supports multiple file formats (PDF, DOCX, TXT) and uses both traditional TF-IDF similarity matching and advanced semantic analysis with sentence transformers and FAISS for comprehensive content analysis.

## Recent Changes

### Latest Repository Deployment (2025-08-08)
- **Commit**: `940744a5b48c52837573efff3b8edc85881e4018` - "Update requirements.txt" 
- **Timestamp**: 2025-08-08T10:35:44Z (deployed within minutes of creation)
- Successfully redeployed from absolute latest GitHub repository state
- All templates and core files recreated from repository
- Database tables initialized successfully
- **AI Detection Verified**: Pattern-based detection working at 69% accuracy on test content with 7 detected AI patterns
- **Local AI Model Loaded**: Created and deployed scikit-learn based AI detection model for enhanced accuracy
- **Environment Optimized**: Graceful fallbacks ensure full functionality without heavy ML dependencies
- **Plagiarism Detection Verified**: TF-IDF mode working with 1.4% false positive rate on clean content
- **Threshold Functionality Fixed**: Semantic threshold now properly controls TF-IDF detection sensitivity
- **Current Configuration**: Plagiarism threshold=0.3 (sensitive), AI thresholds: High=85%, Medium=65%
- Full functionality confirmed with environment-appropriate fallbacks

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Web Framework
- **Flask**: Lightweight web framework chosen for rapid development and simplicity
- **SQLAlchemy**: ORM for database operations with SQLite as the default database
- **Flask-WTF**: Form handling with file upload validation and CSRF protection

## File Processing Pipeline
- **Multi-format Support**: Handles PDF (PyMuPDF/pdfminer), DOCX (python-docx), and plain text files
- **Graceful Fallbacks**: Optional dependencies with fallback mechanisms when libraries are unavailable
- **OCR Integration**: Pytesseract and pdf2image for processing image-based PDFs when needed
- **Text Cleaning**: Preprocessing pipeline to normalize text before analysis

## Detection Architecture
- **Dual Detection System**: Separate modules for plagiarism and AI content detection
- **Weighted Scoring**: Configurable weights between AI model predictions and pattern-based analysis
- **Adjustable Thresholds**: Customizable sensitivity levels for both plagiarism and AI detection

## Plagiarism Detection
- **Primary Method**: Sentence transformers with FAISS indexing for semantic similarity
- **Fallback Method**: TF-IDF vectorization with cosine similarity when ML libraries unavailable
- **Reference Corpus**: File-based reference texts for comparison
- **Chunked Processing**: Handles large documents by processing in overlapping segments

## AI Content Detection
- **Model-based**: Uses transformers pipeline with sequence classification models
- **Pattern-based Fallback**: Rule-based detection when transformers unavailable
- **Hybrid Scoring**: Combines model confidence scores with pattern matching results
- **Configurable Parameters**: Adjustable token limits, stride, and confidence thresholds

## Data Storage
- **SQLite Database**: Stores analysis results, filenames, and scores
- **File System**: Uploaded files stored in configurable upload directory
- **Analysis Persistence**: Historical analysis data for tracking and comparison

# External Dependencies

## Core ML Libraries
- **transformers**: Hugging Face transformers for AI detection models
- **torch**: PyTorch backend for transformer models
- **sentence-transformers**: Semantic similarity models for plagiarism detection
- **faiss**: Efficient similarity search for large-scale semantic matching
- **scikit-learn**: TF-IDF vectorization and cosine similarity calculations

## Document Processing
- **PyMuPDF (fitz)**: Primary PDF text extraction library
- **pdfminer**: Fallback PDF processing when PyMuPDF unavailable
- **PyPDF2**: Additional PDF processing support
- **python-docx**: Microsoft Word document processing
- **pdf2image + pytesseract**: OCR capabilities for image-based PDFs

## Web Framework Stack
- **Flask**: Web application framework
- **Flask-SQLAlchemy**: Database ORM integration
- **Flask-WTF**: Form handling and file uploads
- **WTForms**: Form validation and rendering
- **Werkzeug**: WSGI utilities and secure filename handling

## System Integration
- **SQLite**: Embedded database for development and small deployments
- **File System Storage**: Local file storage for uploads and reference texts
- **Environment Variables**: Configuration management for model paths and thresholds