"""
Entry point for the Offline RAG Application.
Run with: python run.py
"""

import uvicorn
from app.config import settings


def main():
    """Start the FastAPI server."""
    print("=" * 50)
    print("  Offline RAG Application")
    print("=" * 50)
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  Ollama URL: {settings.ollama_base_url}")
    print(f"  PDF Directory: {settings.pdf_dir}")
    print(f"  Vector Store: {settings.vectorstore_dir}")
    print("=" * 50)
    print("\n  Starting server at http://localhost:8000\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
