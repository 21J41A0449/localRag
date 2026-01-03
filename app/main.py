"""
FastAPI Backend for the Agentic RAG Application.
Provides REST API endpoints for PDF upload, querying, and document management.
Features advanced PDF parsing and intelligent query processing.
"""

import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .embeddings import index_documents, load_vectorstore
from .llm import check_ollama_connection

# Try to use advanced ingestion, fallback to basic
try:
    from .advanced_ingestion import process_pdf_advanced, process_all_pdfs_advanced
    USE_ADVANCED = True
except ImportError:
    from .ingestion import load_and_split_pdf, load_all_pdfs, get_pdf_list
    USE_ADVANCED = False

from .ingestion import get_pdf_list

# Try to use agentic pipeline, fallback to basic
try:
    from .agent import get_agentic_rag, reload_agentic_rag
    USE_AGENTIC = True
except ImportError:
    from .rag_chain import get_rag_chain, reload_rag_chain
    USE_AGENTIC = False


# FastAPI app instance
app = FastAPI(
    title="DocuMind AI - Agentic RAG",
    description="Intelligent document Q&A with advanced parsing and agentic reasoning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    include_context: bool = False
    include_reasoning: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    context: Optional[str] = None
    complexity: Optional[str] = None
    reasoning_trace: Optional[List[str]] = None
    sub_queries: Optional[List[str]] = None


class UploadResponse(BaseModel):
    message: str
    files_uploaded: List[str]
    total_chunks: int
    metadata: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    models_available: List[str]
    documents_indexed: bool
    advanced_parsing: bool
    agentic_mode: bool


class DocumentListResponse(BaseModel):
    documents: List[dict]
    total: int


# Store document metadata
document_metadata_store: List[dict] = []


def reindex_all_documents():
    """Background task to reindex all documents with advanced parsing."""
    global document_metadata_store
    
    try:
        if USE_ADVANCED:
            print("Using advanced PDF parsing...")
            all_chunks, all_metadata = process_all_pdfs_advanced()
            document_metadata_store = all_metadata
        else:
            print("Using basic PDF parsing...")
            from .ingestion import load_all_pdfs
            all_chunks = load_all_pdfs()
            document_metadata_store = []
        
        if all_chunks:
            index_documents(all_chunks)
            
            # Reload hybrid retriever for BM25
            try:
                from .hybrid_retrieval import reload_hybrid_retriever
                reload_hybrid_retriever()
                print("Hybrid retriever reloaded")
            except ImportError:
                pass
            
            # Reload the appropriate pipeline
            if USE_AGENTIC:
                reload_agentic_rag()
            else:
                from .rag_chain import reload_rag_chain
                reload_rag_chain()
            
            print(f"Reindexed {len(all_chunks)} chunks from {len(document_metadata_store)} documents")
    except Exception as e:
        print(f"Error during reindexing: {e}")
        import traceback
        traceback.print_exc()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    static_dir = settings.base_dir / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    
    return HTMLResponse(content="<h1>DocuMind AI</h1><p>Static files not found.</p>")


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and capabilities."""
    ollama_status = check_ollama_connection()
    
    # Check if vectorstore exists without reloading (reduces memory usage)
    vectorstore_exists = (settings.vectorstore_dir / "index.faiss").exists()
    
    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_status.get("connected", False),
        models_available=ollama_status.get("models", []),
        documents_indexed=vectorstore_exists,
        advanced_parsing=USE_ADVANCED,
        agentic_mode=USE_AGENTIC
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload PDFs and trigger advanced indexing."""
    uploaded_files = []
    total_chunks = 0
    metadata_list = []
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF"
            )
        
        file_path = settings.pdf_dir / file.filename
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file.filename)
            
            # Process with advanced or basic parsing
            if USE_ADVANCED:
                chunks, metadata = process_pdf_advanced(file_path)
                total_chunks += len(chunks)
                metadata_list.append(metadata)
            else:
                from .ingestion import load_and_split_pdf
                chunks = load_and_split_pdf(file_path)
                total_chunks += len(chunks)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file '{file.filename}': {str(e)}"
            )
        finally:
            file.file.close()
    
    # Trigger background reindexing
    background_tasks.add_task(reindex_all_documents)
    
    return UploadResponse(
        message=f"Files uploaded successfully. {'Advanced' if USE_ADVANCED else 'Basic'} indexing in progress...",
        files_uploaded=uploaded_files,
        total_chunks=total_chunks,
        metadata=metadata_list if USE_ADVANCED else None
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question using agentic processing."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        if USE_AGENTIC:
            agent = get_agentic_rag()
            result = await agent.aprocess_query(request.question)
            
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                complexity=result.get("complexity"),
                reasoning_trace=result.get("reasoning_trace") if request.include_reasoning else None,
                sub_queries=result.get("sub_queries")
            )
        else:
            from .rag_chain import get_rag_chain
            rag_chain = get_rag_chain()
            result = await rag_chain.aquery(request.question)
            
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                context=result["context"] if request.include_context else None
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded PDF documents with metadata."""
    documents = get_pdf_list()
    
    # Enrich with stored metadata if available
    if document_metadata_store:
        for doc in documents:
            for meta in document_metadata_store:
                if meta.get("filename") == doc.get("filename"):
                    doc.update({
                        "pages": meta.get("total_pages"),
                        "has_tables": meta.get("has_tables"),
                        "has_images": meta.get("has_images"),
                        "sections": len(meta.get("sections", []))
                    })
    
    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@app.delete("/documents/{filename}")
async def delete_document(filename: str, background_tasks: BackgroundTasks):
    """Delete a document and reindex."""
    file_path = settings.pdf_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_path.unlink()
        background_tasks.add_task(reindex_all_documents)
        return {"message": f"Document '{filename}' deleted. Reindexing..."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@app.get("/metadata")
async def get_document_metadata():
    """Get detailed metadata for all processed documents."""
    return {
        "documents": document_metadata_store,
        "total": len(document_metadata_store),
        "advanced_parsing": USE_ADVANCED
    }


@app.post("/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    """Manually trigger reindexing."""
    background_tasks.add_task(reindex_all_documents)
    return {"message": f"{'Advanced' if USE_ADVANCED else 'Basic'} reindexing started"}


# Mount static files
static_dir = settings.base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
