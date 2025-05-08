import re
import io
import json
import pdfplumber
import docx
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra whitespace."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_text_from_file(document):
    """Extract text from various file formats"""
    if not document or not hasattr(document, 'name'):
        raise ValueError("Invalid document object")

    file_extension = document.name.split('.')[-1].lower()
    allowed_extensions = ['txt', 'pdf', 'docx', 'json']
    
    if file_extension not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    file_content = document.getvalue()
    
    # Handle different file types
    if file_extension == 'txt':
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                return file_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to decode the text file with supported encodings")
    
    elif file_extension == 'pdf':
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = " ".join([page.extract_text() or "" for page in pdf.pages])
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        return text
    
    elif file_extension == 'docx':
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        if not text:
            raise ValueError("No text could be extracted from the DOCX file")
        return text
    
    elif file_extension == 'json':
        try:
            json_data = json.loads(file_content.decode('utf-8'))
            return json.dumps(json_data, indent=2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {str(e)}")

def split_document(document):
    """Split document into chunks"""
    try:
        text = extract_text_from_file(document)
        
        # Validate extracted text
        if not text:
            raise ValueError("No text could be extracted from the document")
        
        # Clean the text before splitting
        text = clean_text(text)
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=850,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Clean chunks before returning
        return [clean_text(chunk) for chunk in chunks]
    
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise