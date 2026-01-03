"""
Document ingestion module for PDF processing.
Handles loading PDFs and splitting text into chunks with metadata preservation.
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import settings


def load_pdf(file_path: Path) -> List[Document]:
    """
    Load a PDF file and return list of documents (one per page).
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects with page content and metadata
    """
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    
    # Enhance metadata with filename
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_path"] = str(file_path)
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
    
    return chunks


def load_and_split_pdf(file_path: Path) -> List[Document]:
    """
    Load a PDF and split it into chunks in one step.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of chunked Document objects
    """
    documents = load_pdf(file_path)
    return split_documents(documents)


def load_all_pdfs() -> List[Document]:
    """
    Load and process all PDFs in the configured PDF directory.
    
    Returns:
        List of all chunked documents from all PDFs
    """
    all_chunks = []
    pdf_dir = settings.pdf_dir
    
    if not pdf_dir.exists():
        return all_chunks
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            chunks = load_and_split_pdf(pdf_file)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    return all_chunks


def get_pdf_list() -> List[dict]:
    """
    Get list of all uploaded PDFs with metadata.
    
    Returns:
        List of dictionaries with PDF info
    """
    pdf_dir = settings.pdf_dir
    pdfs = []
    
    if not pdf_dir.exists():
        return pdfs
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        stat = pdf_file.stat()
        pdfs.append({
            "filename": pdf_file.name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime
        })
    
    return pdfs
