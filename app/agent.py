"""
Agentic RAG Pipeline Module.
Implements intelligent query processing with decomposition, multi-step retrieval,
self-verification, and answer synthesis.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import settings
from .retrieval import format_context, get_sources
from .embeddings import load_vectorstore
from .llm import get_llm

# Try to use hybrid search, fallback to semantic only
try:
    from .hybrid_retrieval import get_hybrid_retriever, reload_hybrid_retriever, hybrid_search
    USE_HYBRID = True
except ImportError:
    from .retrieval import retrieve_relevant_chunks
    USE_HYBRID = False


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"           # Single fact lookup
    MODERATE = "moderate"       # Multiple related facts
    COMPLEX = "complex"         # Multi-part, comparative, or analytical


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    query: str
    purpose: str
    results: List[Document] = field(default_factory=list)
    answered: bool = False


@dataclass
class AgentState:
    """State tracked throughout the agentic pipeline."""
    original_query: str
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    sub_queries: List[SubQuery] = field(default_factory=list)
    all_context: List[Document] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    verification_passed: bool = False
    final_answer: str = ""
    sources: List[Dict] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 3


class AgenticRAG:
    """
    Agentic RAG pipeline with intelligent query processing.
    
    Features:
    - Query complexity analysis
    - Query decomposition for complex questions
    - Multi-step iterative retrieval
    - Self-verification and re-ranking
    - Answer synthesis with citations
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.vectorstore = load_vectorstore()
        
        # Prompts
        self.ANALYZER_PROMPT = """Analyze the following question and determine its complexity.

Question: {question}

Classify as one of:
- SIMPLE: Single fact lookup (e.g., "What is X?", "When did Y happen?")
- MODERATE: Multiple related facts needed (e.g., "Explain how X works", "What are the steps for Y?")
- COMPLEX: Multi-part, comparative, or analytical (e.g., "Compare X and Y", "What are the pros and cons of X?", questions with multiple sub-questions)

Respond with ONLY one word: SIMPLE, MODERATE, or COMPLEX"""

        self.DECOMPOSER_PROMPT = """Break down this complex question into simpler sub-questions that can be answered individually.

Original Question: {question}

Generate 2-4 sub-questions that together would help answer the original question completely.
Format each sub-question on a new line, starting with a number and period.

Sub-questions:"""

        self.VERIFIER_PROMPT = """Review if the provided context adequately answers the question.

Question: {question}

Context provided:
{context}

Does the context contain enough information to accurately answer the question?
Respond with ONLY: YES or NO

If NO, what specific information is missing? (one line)"""

        self.SYNTHESIS_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If information is missing, say "I don't know based on the provided documents."
3. Never hallucinate or invent information.
4. Cite sources by mentioning the file name and page number when possible.
5. For complex questions, structure your answer clearly.

CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive, well-structured answer:"""
    
    def reload_vectorstore(self):
        """Reload the vector store from disk."""
        self.vectorstore = load_vectorstore()
    
    def _analyze_complexity(self, question: str) -> QueryComplexity:
        """Analyze query complexity using LLM."""
        try:
            prompt = self.ANALYZER_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            result = response.content.strip().upper()
            
            if "COMPLEX" in result:
                return QueryComplexity.COMPLEX
            elif "MODERATE" in result:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
        except Exception as e:
            print(f"Complexity analysis failed: {e}")
            return QueryComplexity.SIMPLE
    
    def _decompose_query(self, question: str) -> List[SubQuery]:
        """Decompose complex query into sub-queries."""
        try:
            prompt = self.DECOMPOSER_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            
            sub_queries = []
            lines = response.content.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove numbering
                    query = line.lstrip("0123456789.):- ").strip()
                    if query:
                        sub_queries.append(SubQuery(
                            query=query,
                            purpose=f"Answer part of: {question[:50]}..."
                        ))
            
            # Always include original query as final sub-query
            if sub_queries:
                return sub_queries
            else:
                return [SubQuery(query=question, purpose="Direct answer")]
                
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [SubQuery(query=question, purpose="Direct answer")]
    
    def _retrieve_for_query(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve documents using hybrid search (semantic + keyword)."""
        k = k or settings.top_k
        
        if USE_HYBRID:
            # Use hybrid search combining FAISS + BM25
            try:
                retriever = get_hybrid_retriever()
                return retriever.search(query, k)
            except Exception as e:
                print(f"Hybrid search failed: {e}, falling back to semantic")
        
        # Fallback to semantic-only search
        if self.vectorstore is None:
            return []
        
        from .retrieval import retrieve_relevant_chunks
        return retrieve_relevant_chunks(query, self.vectorstore, k)
    
    def _verify_context(self, question: str, context: str) -> Tuple[bool, str]:
        """Verify if context adequately answers the question."""
        try:
            prompt = self.VERIFIER_PROMPT.format(question=question, context=context[:3000])
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            is_adequate = "YES" in result.upper().split("\n")[0]
            missing_info = ""
            
            if not is_adequate and len(result.split("\n")) > 1:
                missing_info = result.split("\n")[1].strip()
            
            return is_adequate, missing_info
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return True, ""  # Assume adequate on error
    
    def _synthesize_answer(self, question: str, context: str) -> str:
        """Generate final answer from context."""
        prompt = self.SYNTHESIS_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def _deduplicate_docs(self, docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate documents based on content."""
        seen_content = set()
        unique = []
        
        for doc, score in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique.append((doc, score))
        
        return unique
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the agentic pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, reasoning trace
        """
        if self.vectorstore is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload PDF files first.",
                "sources": [],
                "reasoning_trace": ["No vector store available"],
                "complexity": "N/A"
            }
        
        # Initialize state
        state = AgentState(original_query=question)
        state.reasoning_trace.append(f"Received query: {question}")
        
        # Step 1: Analyze complexity
        state.complexity = self._analyze_complexity(question)
        state.reasoning_trace.append(f"Query complexity: {state.complexity.value}")
        
        # Step 2: Decompose if complex
        if state.complexity == QueryComplexity.COMPLEX:
            state.sub_queries = self._decompose_query(question)
            state.reasoning_trace.append(f"Decomposed into {len(state.sub_queries)} sub-queries")
        else:
            state.sub_queries = [SubQuery(query=question, purpose="Direct answer")]
        
        # Step 3: Multi-step retrieval
        all_results = []
        for sq in state.sub_queries:
            results = self._retrieve_for_query(sq.query)
            sq.results = [doc for doc, _ in results]
            all_results.extend(results)
            state.reasoning_trace.append(f"Retrieved {len(results)} chunks for: {sq.query[:50]}...")
        
        # Deduplicate and re-rank
        all_results = self._deduplicate_docs(all_results)
        all_results.sort(key=lambda x: x[1])  # Sort by score (lower is better for FAISS)
        
        # Take top results
        top_results = all_results[:settings.top_k * 2]
        state.all_context = [doc for doc, _ in top_results]
        
        if not state.all_context:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "reasoning_trace": state.reasoning_trace,
                "complexity": state.complexity.value
            }
        
        # Format context
        context = format_context(top_results)
        state.sources = get_sources(top_results)
        
        # Step 4: Verification loop
        while state.iterations < state.max_iterations:
            state.iterations += 1
            
            is_adequate, missing = self._verify_context(question, context)
            
            if is_adequate:
                state.verification_passed = True
                state.reasoning_trace.append("Verification passed - context is adequate")
                break
            else:
                state.reasoning_trace.append(f"Verification failed - missing: {missing}")
                
                # Try to retrieve more with refined query
                if missing:
                    additional = self._retrieve_for_query(f"{question} {missing}", k=2)
                    if additional:
                        all_results.extend(additional)
                        all_results = self._deduplicate_docs(all_results)
                        all_results.sort(key=lambda x: x[1])
                        top_results = all_results[:settings.top_k * 2]
                        context = format_context(top_results)
                        state.sources = get_sources(top_results)
                        state.reasoning_trace.append(f"Retrieved additional context for: {missing[:30]}...")
                else:
                    break
        
        # Step 5: Synthesize answer
        state.final_answer = self._synthesize_answer(question, context)
        state.reasoning_trace.append("Generated final answer")
        
        return {
            "answer": state.final_answer,
            "sources": state.sources,
            "reasoning_trace": state.reasoning_trace,
            "complexity": state.complexity.value,
            "sub_queries": [sq.query for sq in state.sub_queries] if state.complexity == QueryComplexity.COMPLEX else []
        }
    
    async def aprocess_query(self, question: str) -> Dict[str, Any]:
        """Async version of process_query."""
        # For now, just call sync version
        # Can be optimized with async LLM calls later
        return self.process_query(question)


# Global instance
_agentic_rag: Optional[AgenticRAG] = None


def get_agentic_rag() -> AgenticRAG:
    """Get or create the global AgenticRAG instance."""
    global _agentic_rag
    
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG()
    
    return _agentic_rag


def reload_agentic_rag() -> AgenticRAG:
    """Force reload the AgenticRAG with fresh vector store."""
    global _agentic_rag
    _agentic_rag = AgenticRAG()
    return _agentic_rag
