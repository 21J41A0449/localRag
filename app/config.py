"""
Configuration module for the Offline RAG Application.
All settings are configurable via environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    llm_model: str = Field(default="qwen2.5:3b", description="Ollama LLM model name")
    embedding_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    
    # Chunking Configuration
    chunk_size: int = Field(default=800, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=150, description="Overlap between chunks")
    
    # Retrieval Configuration
    top_k: int = Field(default=4, description="Number of chunks to retrieve")
    
    # LLM Parameters
    temperature: float = Field(default=0.0, description="LLM temperature (0 = deterministic)")
    
    # Path Configuration
    base_dir: Path = Field(default=Path(__file__).parent.parent, description="Base directory")
    
    @property
    def data_dir(self) -> Path:
        """Directory for storing data (PDFs and vector store)."""
        return self.base_dir / "data"
    
    @property
    def pdf_dir(self) -> Path:
        """Directory for storing uploaded PDFs."""
        return self.data_dir / "pdfs"
    
    @property
    def vectorstore_dir(self) -> Path:
        """Directory for storing FAISS index."""
        return self.data_dir / "vectorstore"
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        extra = "ignore"


# Global settings instance
settings = Settings()
settings.ensure_directories()
