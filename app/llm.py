"""
LLM module for Ollama integration.
Handles LLM initialization and model management.
"""

from typing import Optional
from langchain_ollama import ChatOllama

from .config import settings


# Global LLM instance
_llm: Optional[ChatOllama] = None


def get_llm(model: Optional[str] = None) -> ChatOllama:
    """
    Get or create the LLM instance.
    
    Args:
        model: Optional model name override
        
    Returns:
        ChatOllama instance
    """
    global _llm
    
    model_name = model or settings.llm_model
    
    # Create new instance if model changed or doesn't exist
    if _llm is None or (model and model != settings.llm_model):
        _llm = ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature
        )
    
    return _llm


def check_ollama_connection() -> dict:
    """
    Check if Ollama is running and models are available.
    
    Returns:
        Dictionary with connection status and available models
    """
    import httpx
    
    try:
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=5.0
        )
        
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            return {
                "connected": True,
                "models": models,
                "llm_model_available": settings.llm_model in models or any(
                    settings.llm_model in m for m in models
                ),
                "embedding_model_available": settings.embedding_model in models or any(
                    settings.embedding_model in m for m in models
                )
            }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "models": []
        }
    
    return {"connected": False, "models": []}
