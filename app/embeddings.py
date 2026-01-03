"""
Embeddings and Vector Store module.
Handles embedding generation using Ollama and FAISS vector store management.
"""

from pathlib import Path
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import settings


# Global embeddings instance
_embeddings: Optional[OllamaEmbeddings] = None


def get_embeddings() -> OllamaEmbeddings:
    """
    Get or create the embeddings instance.
    
    Returns:
        OllamaEmbeddings instance
    """
    global _embeddings
    
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
    
    return _embeddings


def create_vectorstore(documents: List[Document]) -> FAISS:
    """
    Create a new FAISS vector store from documents.
    
    Args:
        documents: List of documents to index
        
    Returns:
        FAISS vector store instance
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Persist vector store to disk.
    
    Args:
        vectorstore: FAISS vector store to save
    """
    vectorstore_path = settings.vectorstore_dir
    vectorstore.save_local(str(vectorstore_path))
    print(f"Vector store saved to {vectorstore_path}")


def load_vectorstore() -> Optional[FAISS]:
    """
    Load vector store from disk if it exists.
    
    Returns:
        FAISS vector store or None if not found
    """
    vectorstore_path = settings.vectorstore_dir
    index_file = vectorstore_path / "index.faiss"
    
    if not index_file.exists():
        print("No existing vector store found")
        return None
    
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded vector store from {vectorstore_path}")
        return vectorstore
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def index_documents(documents: List[Document]) -> FAISS:
    """
    Index documents: create vector store and persist to disk.
    
    Args:
        documents: List of documents to index
        
    Returns:
        FAISS vector store instance
    """
    if not documents:
        raise ValueError("No documents to index")
    
    print(f"Indexing {len(documents)} document chunks...")
    vectorstore = create_vectorstore(documents)
    save_vectorstore(vectorstore)
    print("Indexing complete!")
    
    return vectorstore


def add_documents_to_store(
    vectorstore: FAISS,
    documents: List[Document]
) -> FAISS:
    """
    Add new documents to an existing vector store.
    
    Args:
        vectorstore: Existing FAISS vector store
        documents: New documents to add
        
    Returns:
        Updated FAISS vector store
    """
    vectorstore.add_documents(documents)
    save_vectorstore(vectorstore)
    return vectorstore
