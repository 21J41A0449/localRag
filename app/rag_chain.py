"""
RAG Chain module - Core pipeline for retrieval-augmented generation.
Implements strict grounding to prevent hallucination.
"""

from typing import Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .retrieval import retrieve_relevant_chunks, format_context, get_sources
from .embeddings import load_vectorstore


# MANDATORY SYSTEM PROMPT - Enforces strict grounding
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES - YOU MUST FOLLOW THESE:
1. Answer ONLY from the provided context below. Do not use any external knowledge.
2. If the information to answer the question is NOT in the context, respond exactly: "I don't know based on the provided documents."
3. Never make assumptions or infer information that is not explicitly stated.
4. Never hallucinate or invent information.
5. If you're unsure, say "I don't know based on the provided documents."
6. When possible, cite the source (file name and page number) of your answer.
7. Keep answers concise and directly relevant to the question.

CONTEXT FROM UPLOADED DOCUMENTS:
{context}
"""

HUMAN_PROMPT = """Question: {question}

Remember: Answer ONLY from the context above. If the answer is not in the context, say "I don't know based on the provided documents."

Answer:"""


class RAGChain:
    """
    RAG Chain for question answering with strict grounding.
    """
    
    def __init__(self, vectorstore: Optional[FAISS] = None):
        """
        Initialize the RAG chain.
        
        Args:
            vectorstore: Optional pre-loaded vector store
        """
        self.vectorstore = vectorstore or load_vectorstore()
        self.llm = get_llm()
        self._setup_chain()
    
    def _setup_chain(self) -> None:
        """Set up the prompt template and chain."""
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
        ])
    
    def reload_vectorstore(self) -> bool:
        """
        Reload the vector store from disk.
        
        Returns:
            True if reload successful, False otherwise
        """
        self.vectorstore = load_vectorstore()
        return self.vectorstore is not None
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return grounded answer with sources.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and context
        """
        if self.vectorstore is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload PDF files first.",
                "sources": [],
                "context": ""
            }
        
        # Retrieve relevant chunks
        results = retrieve_relevant_chunks(question, self.vectorstore)
        
        if not results:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "context": ""
            }
        
        # Format context and extract sources
        context = format_context(results)
        sources = get_sources(results)
        
        # Generate answer using LLM
        messages = self.prompt.format_messages(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }
    
    async def aquery(self, question: str) -> Dict[str, Any]:
        """
        Async version of query for FastAPI.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and context
        """
        if self.vectorstore is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload PDF files first.",
                "sources": [],
                "context": ""
            }
        
        # Retrieve relevant chunks
        results = retrieve_relevant_chunks(question, self.vectorstore)
        
        if not results:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "context": ""
            }
        
        # Format context and extract sources
        context = format_context(results)
        sources = get_sources(results)
        
        # Generate answer using LLM
        messages = self.prompt.format_messages(
            context=context,
            question=question
        )
        
        response = await self.llm.ainvoke(messages)
        answer = response.content
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }


# Global RAG chain instance
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """
    Get or create the global RAG chain instance.
    
    Returns:
        RAGChain instance
    """
    global _rag_chain
    
    if _rag_chain is None:
        _rag_chain = RAGChain()
    
    return _rag_chain


def reload_rag_chain() -> RAGChain:
    """
    Force reload the RAG chain with fresh vector store.
    
    Returns:
        New RAGChain instance
    """
    global _rag_chain
    _rag_chain = RAGChain()
    return _rag_chain
