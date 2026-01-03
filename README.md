# 🧠 DocuMind AI - Agentic RAG

A production-ready **offline RAG (Retrieval-Augmented Generation)** application with intelligent agentic reasoning, hybrid search, and advanced PDF parsing.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Agentic RAG** | Query decomposition, multi-step retrieval, self-verification |
| 🔍 **Hybrid Search** | Combines FAISS semantic + BM25 keyword search with RRF |
| 📄 **Advanced PDF Parsing** | Tables, images, flowcharts via Unstructured |
| 🔒 **100% Offline** | Works without internet using Ollama |
| 📍 **Source Citations** | Every answer includes page references |
| 🎨 **Modern UI** | Premium dark theme with glassmorphism |

## 📸 Screenshot

![DocuMind AI Interface](https://via.placeholder.com/800x450?text=DocuMind+AI+Interface)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              🧠 Agentic Pipeline                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Analyze  │→ │Decompose │→ │ Retrieve │→ │ Verify   │    │
│  │Complexity│  │ Query    │  │ (Hybrid) │  │ Context  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              🔍 Hybrid Retrieval                             │
│  ┌─────────────────┐     ┌─────────────────┐                │
│  │  FAISS Semantic │ ──► │  Reciprocal     │                │
│  │     Search      │     │  Rank Fusion    │ ──► Results    │
│  │  BM25 Keyword   │ ──► │                 │                │
│  └─────────────────┘     └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed

### 1. Install Ollama Models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Clone & Install

```bash
git clone https://github.com/21J41A0449/localRag.git
cd localRag

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```bash
python run.py
```

Open http://localhost:8000 in your browser.

## 📁 Project Structure

```
localRag/
├── app/
│   ├── agent.py              # Agentic RAG pipeline
│   ├── hybrid_retrieval.py   # BM25 + FAISS hybrid search
│   ├── advanced_ingestion.py # Unstructured PDF parsing
│   ├── embeddings.py         # FAISS vector store
│   ├── retrieval.py          # Semantic search
│   ├── rag_chain.py          # Basic RAG chain
│   ├── llm.py                # Ollama integration
│   ├── config.py             # Configuration
│   └── main.py               # FastAPI backend
├── static/
│   ├── index.html            # Web interface
│   ├── styles.css            # Premium styling
│   └── app.js                # Frontend logic
├── data/
│   ├── pdfs/                 # Uploaded documents
│   └── vectorstore/          # FAISS index
├── requirements.txt
├── run.py                    # Entry point
└── README.md
```

## 🔧 Configuration

Edit `app/config.py` or use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LLM_MODEL` | `qwen2.5:3b` | Ollama LLM model |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_CHUNK_SIZE` | `800` | Text chunk size |
| `RAG_TOP_K` | `4` | Retrieved chunks |
| `RAG_TEMPERATURE` | `0` | LLM temperature |

### Switch Models

```bash
# Use smaller model for low RAM
ollama pull phi3:mini
# Then update RAG_LLM_MODEL=phi3:mini
```

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/upload` | POST | Upload PDFs |
| `/query` | POST | Ask question |
| `/documents` | GET | List documents |
| `/documents/{name}` | DELETE | Delete document |
| `/health` | GET | Health check |
| `/metadata` | GET | Document metadata |

## 💡 How It Works

### Agentic Pipeline

1. **Query Analysis** - Detects complexity (simple/moderate/complex)
2. **Decomposition** - Breaks complex questions into sub-queries
3. **Hybrid Retrieval** - Searches using both semantic and keyword methods
4. **Verification** - Checks if context answers the question
5. **Synthesis** - Generates grounded answer with citations

### Hybrid Search

Combines two retrieval methods using Reciprocal Rank Fusion:
- **FAISS** (Semantic) - Finds conceptually similar content
- **BM25** (Keyword) - Finds exact term matches

## 🤝 Contributing

Pull requests welcome! Please read contributing guidelines first.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

Built with ❤️ using LangChain, Ollama, and FastAPI
