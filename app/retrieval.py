"""
Retrieval module for semantic search.
Handles querying the vector store and formatting results.
"""

from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import settings
from .embeddings import load_vectorstore


def retrieve_relevant_chunks(
    query: str,
    vectorstore: FAISS,
    k: Optional[int] = None
) -> List[Tuple[Document, float]]:
    """
    Retrieve most relevant document chunks for a query.
    
    Args:
        query: User's question
        vectorstore: FAISS vector store to search
        k: Number of results to return (default from settings)
        
    Returns:
        List of (Document, score) tuples sorted by relevance
    """
    k = k or settings.top_k
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def format_context(results: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        results: List of (Document, score) tuples
        
    Returns:
        Formatted context string with source citations
    """
    if not results:
        return "No relevant context found."
    
    context_parts = []
    
    for idx, (doc, score) in enumerate(results, 1):
        source_file = doc.metadata.get("source_file", "Unknown")
        page_num = doc.metadata.get("page", "Unknown")
        
        context_parts.append(
            f"[Source {idx}: {source_file}, Page {page_num + 1 if isinstance(page_num, int) else page_num}]\n"
            f"{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def get_sources(results: List[Tuple[Document, float]]) -> List[dict]:
    """
    Extract source information from retrieval results.
    
    Args:
        results: List of (Document, score) tuples
        
    Returns:
        List of source dictionaries with file and page info
    """
    sources = []
    seen = set()
    
    for doc, score in results:
        source_file = doc.metadata.get("source_file", "Unknown")
        page_num = doc.metadata.get("page", 0)
        
        # Create unique key to avoid duplicates
        key = f"{source_file}:{page_num}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source_file,
                "page": page_num + 1 if isinstance(page_num, int) else page_num,
                "relevance_score": round(float(score), 4)
            })
    
    return sources
