"""
Hybrid Retrieval Module.
Combines semantic search (FAISS) with keyword search (BM25) using Reciprocal Rank Fusion.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import settings
from .embeddings import load_vectorstore


@dataclass
class SearchResult:
    """Represents a search result with scores from different methods."""
    document: Document
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    semantic_rank: int = 0
    keyword_rank: int = 0


class BM25Retriever:
    """
    BM25 keyword-based retriever for sparse search.
    Implements Okapi BM25 algorithm locally without external dependencies.
    """
    
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of documents to index
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Build index
        self.doc_lengths = []
        self.doc_tokens = []
        self.doc_freqs = defaultdict(int)  # Document frequency per term
        self.term_freqs = []  # Term frequency per document
        
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace/punctuation."""
        import re
        # Convert to lowercase and extract word tokens
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return tokens
    
    def _build_index(self):
        """Build the BM25 index from documents."""
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies for this document
            tf = defaultdict(int)
            unique_terms = set()
            for token in tokens:
                tf[token] += 1
                unique_terms.add(token)
            
            self.term_freqs.append(tf)
            
            # Update document frequency
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
        self.num_docs = len(self.documents)
    
    def _calculate_idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document given query tokens."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.term_freqs[doc_idx]
        
        for token in query_tokens:
            if token not in tf_dict:
                continue
            
            tf = tf_dict[token]
            idf = self._calculate_idf(token)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples sorted by score descending
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate BM25 scores for all documents
        scores = []
        for idx in range(len(self.documents)):
            score = self._calculate_bm25_score(query_tokens, idx)
            if score > 0:
                scores.append((self.documents[idx], score, idx))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return [(doc, score) for doc, score, _ in scores[:k]]


class HybridRetriever:
    """
    Hybrid retriever combining semantic (FAISS) and keyword (BM25) search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
        documents: Optional[List[Document]] = None,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vectorstore: FAISS vector store for semantic search
            documents: Documents for BM25 keyword search
            semantic_weight: Weight for semantic search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            rrf_k: RRF constant (typically 60)
        """
        self.vectorstore = vectorstore or load_vectorstore()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        # Initialize BM25 if documents provided
        self.bm25 = None
        if documents:
            self.bm25 = BM25Retriever(documents)
        elif self.vectorstore:
            # Extract documents from vectorstore
            self._init_bm25_from_vectorstore()
    
    def _init_bm25_from_vectorstore(self):
        """Initialize BM25 from vectorstore documents."""
        try:
            # Get all documents from FAISS
            # This is a workaround since FAISS doesn't expose documents directly
            docstore = self.vectorstore.docstore
            index_to_id = self.vectorstore.index_to_docstore_id
            
            documents = []
            for idx in range(len(index_to_id)):
                doc_id = index_to_id[idx]
                doc = docstore.search(doc_id)
                if doc:
                    documents.append(doc)
            
            if documents:
                self.bm25 = BM25Retriever(documents)
                print(f"Initialized BM25 with {len(documents)} documents")
        except Exception as e:
            print(f"Could not initialize BM25 from vectorstore: {e}")
            self.bm25 = None
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]]
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF Score = sum(1 / (k + rank)) for each result list
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            
        Returns:
            Combined and re-ranked results
        """
        # Create lookup by document content hash
        results_map: Dict[int, SearchResult] = {}
        
        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in results_map:
                results_map[content_hash] = SearchResult(document=doc)
            
            results_map[content_hash].semantic_score = score
            results_map[content_hash].semantic_rank = rank
            # RRF contribution from semantic search
            results_map[content_hash].hybrid_score += self.semantic_weight * (1 / (self.rrf_k + rank))
        
        # Process keyword results
        for rank, (doc, score) in enumerate(keyword_results, 1):
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in results_map:
                results_map[content_hash] = SearchResult(document=doc)
            
            results_map[content_hash].keyword_score = score
            results_map[content_hash].keyword_rank = rank
            # RRF contribution from keyword search
            results_map[content_hash].hybrid_score += self.keyword_weight * (1 / (self.rrf_k + rank))
        
        # Sort by hybrid score descending
        results = list(results_map.values())
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return results
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining semantic and keyword methods.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or settings.top_k
        
        # Get more results from each method for better fusion
        fetch_k = k * 3
        
        # Semantic search (FAISS)
        semantic_results = []
        if self.vectorstore:
            try:
                semantic_results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
            except Exception as e:
                print(f"Semantic search failed: {e}")
        
        # Keyword search (BM25)
        keyword_results = []
        if self.bm25:
            try:
                keyword_results = self.bm25.search(query, k=fetch_k)
            except Exception as e:
                print(f"Keyword search failed: {e}")
        
        # If only one method available, return its results
        if not semantic_results and not keyword_results:
            return []
        
        if not keyword_results:
            return semantic_results[:k]
        
        if not semantic_results:
            return keyword_results[:k]
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        
        # Return top k as (Document, score) tuples
        return [(r.document, r.hybrid_score) for r in combined[:k]]
    
    def search_with_details(self, query: str, k: int = None) -> List[SearchResult]:
        """
        Perform hybrid search and return detailed results with all scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of SearchResult objects with detailed scores
        """
        k = k or settings.top_k
        fetch_k = k * 3
        
        semantic_results = []
        if self.vectorstore:
            try:
                semantic_results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
            except Exception as e:
                print(f"Semantic search failed: {e}")
        
        keyword_results = []
        if self.bm25:
            try:
                keyword_results = self.bm25.search(query, k=fetch_k)
            except Exception as e:
                print(f"Keyword search failed: {e}")
        
        if not semantic_results and not keyword_results:
            return []
        
        combined = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        return combined[:k]


# Global hybrid retriever instance
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the global hybrid retriever."""
    global _hybrid_retriever
    
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    
    return _hybrid_retriever


def reload_hybrid_retriever() -> HybridRetriever:
    """Force reload the hybrid retriever."""
    global _hybrid_retriever
    _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def hybrid_search(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Convenience function for hybrid search.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of (Document, score) tuples
    """
    retriever = get_hybrid_retriever()
    return retriever.search(query, k)
