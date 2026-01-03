# Offline RAG Application

A production-ready offline Retrieval-Augmented Generation (RAG) application that answers questions exclusively from uploaded PDF documents using local Ollama models.

## Features

- 📄 **PDF Upload**: Upload and process multiple PDF files
- 🔍 **Semantic Search**: Find relevant content using vector similarity
- 🤖 **Local LLM**: Powered by Ollama (qwen2.5:3b by default)
- 🔒 **Completely Offline**: No internet required after setup
- 📍 **Source Citations**: Answers include page references
- ⚡ **Fast Responses**: Optimized for 8-16GB RAM machines

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running

### Install Ollama Models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Installation

1. **Clone the repository**
```bash
cd d:\offlinerag
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**
```bash
python run.py
```

2. **Open browser**
Navigate to `http://localhost:8000`

3. **Upload PDFs**
Drag and drop PDF files or click to browse

4. **Ask questions**
Type your question and get answers grounded in your documents

## Configuration

Environment variables (optional):
- `RAG_LLM_MODEL`: LLM model name (default: `qwen2.5:3b`)
- `RAG_EMBEDDING_MODEL`: Embedding model (default: `nomic-embed-text`)
- `RAG_CHUNK_SIZE`: Text chunk size (default: `800`)
- `RAG_CHUNK_OVERLAP`: Chunk overlap (default: `150`)
- `RAG_TOP_K`: Number of chunks to retrieve (default: `4`)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/upload` | POST | Upload PDF files |
| `/query` | POST | Ask a question |
| `/documents` | GET | List documents |
| `/documents/{name}` | DELETE | Remove document |
| `/health` | GET | Health check |

## Project Structure

```
offlinerag/
├── app/
│   ├── config.py      # Configuration
│   ├── ingestion.py   # PDF processing
│   ├── embeddings.py  # Vector store
│   ├── retrieval.py   # Search
│   ├── llm.py         # Ollama LLM
│   ├── rag_chain.py   # RAG pipeline
│   └── main.py        # FastAPI app
├── static/            # Web frontend
├── data/
│   ├── pdfs/          # Uploaded files
│   └── vectorstore/   # FAISS index
├── requirements.txt
└── run.py             # Entry point
```

## Switching Models

Edit `app/config.py` or set environment variable:
```bash
set RAG_LLM_MODEL=phi3:mini  # Windows
export RAG_LLM_MODEL=phi3:mini  # Linux/Mac
```

Compatible models: `qwen2.5:3b`, `phi3:mini`, `llama3.2:3b`, `mistral:7b`

## License

MIT License
