"""
Advanced Document Ingestion Module using Unstructured.
Handles complex PDFs with tables, images, flowcharts, and structured content.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def extract_with_unstructured(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract elements from PDF using Unstructured library.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of extracted elements with metadata
    """
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",  # High resolution for better extraction
            extract_images_in_pdf=True,
            infer_table_structure=True,
            include_page_breaks=True,
        )
        
        extracted = []
        current_section = "Introduction"
        
        for element in elements:
            elem_type = type(element).__name__
            
            # Track section headers
            if elem_type == "Title":
                current_section = str(element)
            
            # Get element metadata
            metadata = {
                "element_type": elem_type,
                "page_number": getattr(element.metadata, 'page_number', 1),
                "section": current_section,
                "source_file": file_path.name,
            }
            
            # Extract text content
            text_content = str(element)
            
            # Handle tables specially - convert to markdown
            if elem_type == "Table":
                text_content = _table_to_markdown(element)
                metadata["is_table"] = True
            
            # Handle images - get caption/description
            if elem_type == "Image":
                text_content = f"[Image: {getattr(element, 'text', 'Visual content')}]"
                metadata["is_image"] = True
            
            if text_content.strip():
                extracted.append({
                    "content": text_content,
                    "metadata": metadata
                })
        
        return extracted
        
    except ImportError:
        print("Unstructured not available, falling back to basic extraction")
        return _fallback_extraction(file_path)
    except Exception as e:
        print(f"Unstructured extraction failed: {e}, using fallback")
        return _fallback_extraction(file_path)


def _table_to_markdown(table_element) -> str:
    """Convert table element to markdown format."""
    try:
        html = getattr(table_element.metadata, 'text_as_html', None)
        if html:
            # Simple HTML table to markdown conversion
            return f"[Table Content]\n{str(table_element)}"
        return str(table_element)
    except:
        return str(table_element)


def _fallback_extraction(file_path: Path) -> List[Dict[str, Any]]:
    """Fallback to PyPDF if Unstructured fails."""
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    
    extracted = []
    for page in pages:
        extracted.append({
            "content": page.page_content,
            "metadata": {
                "element_type": "NarrativeText",
                "page_number": page.metadata.get("page", 0) + 1,
                "section": "Document",
                "source_file": file_path.name,
            }
        })
    
    return extracted


def generate_document_metadata(file_path: Path, elements: List[Dict]) -> Dict[str, Any]:
    """
    Generate rich metadata for the document.
    
    Args:
        file_path: Path to the PDF
        elements: Extracted elements
        
    Returns:
        Document metadata dictionary
    """
    stat = file_path.stat()
    
    # Count element types
    element_types = {}
    sections = set()
    total_pages = 1
    
    for elem in elements:
        elem_type = elem["metadata"]["element_type"]
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
        sections.add(elem["metadata"]["section"])
        total_pages = max(total_pages, elem["metadata"]["page_number"])
    
    return {
        "filename": file_path.name,
        "file_size": stat.st_size,
        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "total_pages": total_pages,
        "total_elements": len(elements),
        "element_counts": element_types,
        "sections": list(sections),
        "has_tables": element_types.get("Table", 0) > 0,
        "has_images": element_types.get("Image", 0) > 0,
    }


def create_enhanced_chunks(
    elements: List[Dict[str, Any]],
    doc_metadata: Dict[str, Any],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Create enhanced document chunks with rich metadata.
    
    Args:
        elements: Extracted elements
        doc_metadata: Document-level metadata
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with enhanced metadata
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for idx, element in enumerate(elements):
        content = element["content"]
        elem_metadata = element["metadata"]
        
        # For short elements, don't split
        if len(content) <= chunk_size:
            chunk_metadata = {
                **elem_metadata,
                "chunk_index": len(all_chunks),
                "document_title": doc_metadata.get("filename", "Unknown"),
                "total_pages": doc_metadata.get("total_pages", 1),
            }
            
            all_chunks.append(Document(
                page_content=content,
                metadata=chunk_metadata
            ))
        else:
            # Split longer elements
            texts = text_splitter.split_text(content)
            for i, text in enumerate(texts):
                chunk_metadata = {
                    **elem_metadata,
                    "chunk_index": len(all_chunks),
                    "sub_chunk": i,
                    "document_title": doc_metadata.get("filename", "Unknown"),
                    "total_pages": doc_metadata.get("total_pages", 1),
                }
                
                all_chunks.append(Document(
                    page_content=text,
                    metadata=chunk_metadata
                ))
    
    return all_chunks


def process_pdf_advanced(file_path: Path) -> tuple[List[Document], Dict[str, Any]]:
    """
    Process a PDF with advanced extraction.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (chunks, document_metadata)
    """
    print(f"Processing {file_path.name} with advanced extraction...")
    
    # Extract elements
    elements = extract_with_unstructured(file_path)
    print(f"  Extracted {len(elements)} elements")
    
    # Generate document metadata
    doc_metadata = generate_document_metadata(file_path, elements)
    print(f"  Pages: {doc_metadata['total_pages']}, Tables: {doc_metadata['has_tables']}, Images: {doc_metadata['has_images']}")
    
    # Create enhanced chunks
    chunks = create_enhanced_chunks(elements, doc_metadata)
    print(f"  Created {len(chunks)} chunks")
    
    return chunks, doc_metadata


def process_all_pdfs_advanced() -> tuple[List[Document], List[Dict[str, Any]]]:
    """
    Process all PDFs in the configured directory with advanced extraction.
    
    Returns:
        Tuple of (all_chunks, all_metadata)
    """
    all_chunks = []
    all_metadata = []
    
    pdf_dir = settings.pdf_dir
    if not pdf_dir.exists():
        return all_chunks, all_metadata
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            chunks, metadata = process_pdf_advanced(pdf_file)
            all_chunks.extend(chunks)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    return all_chunks, all_metadata
