# 🤖 Rackhost Knowledge Base RAG System

> A production-ready Retrieval-Augmented Generation (RAG) system for automated Hungarian customer support, built with ChromaDB, Ollama, and SentenceTransformers.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.ai/)

## 📋 Overview

This project implements an end-to-end RAG pipeline that transforms Rackhost's web-based knowledge base into an intelligent Q&A system. The system can answer customer support questions in Hungarian by retrieving relevant documentation and generating contextual responses.

### Key Features

- 🌐 **Automated Web Scraping**: Extracts content from Rackhost's knowledge base with pagination support
- 📊 **Smart Chunking**: Implements overlapping text chunks for optimal context preservation  
- 🔍 **Semantic Search**: Uses vector embeddings for accurate document retrieval
- 🇭🇺 **Hungarian Language Support**: Optimized prompts and models for Hungarian text generation
- 💻 **Local-First**: Runs entirely on-device (Mac M2 optimized) with no API costs
- 🎯 **High Accuracy**: Retrieves relevant context and generates coherent 2-4 sentence answers

## 🏗️ Architecture

```
Web Scraping → Data Cleaning → Chunking → Vector Embedding → ChromaDB → RAG Query
     ↓              ↓              ↓              ↓              ↓          ↓
  scraper.py   build_kb_clean  chunk_kb.py   build_index.py  chroma_kb  rag_qa_ollama.py
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Scraping** | BeautifulSoup, Requests | Extract KB articles from website |
| **Data Processing** | Python, JSON | Clean and structure raw HTML |
| **Chunking** | Custom algorithm | Split documents with 200-char overlap |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Convert text to vectors |
| **Vector Database** | ChromaDB | Store and query embeddings |
| **LLM** | Ollama (Mistral 7B) | Generate natural language responses |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Ollama installed ([ollama.ai](https://ollama.ai))
- 8GB+ RAM (for local LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rackhostllm.git
cd rackhostllm

# Install Python dependencies
pip install -r requirements.txt

# Install and start Ollama
brew install ollama
brew services start ollama

# Pull the Mistral model
ollama pull mistral:latest
```

### Usage

#### 1. Scrape Knowledge Base (Optional - data included)

```bash
python scripts/scraper.py
```

#### 2. Process and Chunk Data

```bash
# Clean the raw export
python scripts/build_kb_clean.py

# Create overlapping chunks
python scripts/chunk_kb.py
```

#### 3. Build Vector Index

```bash
python scripts/build_index.py
```

#### 4. Query the System

```bash
# Ask a question in Hungarian
python rag/rag_qa_ollama.py "Hogyan állítsam be a domain-t?"

# Example output:
# 🔍 Keresés a tudásbázisban: 'Hogyan állítsam be a domain-t?'
#
# ✓ 2 releváns dokumentum találva
#   • Domain beállítások (távolság: 0.234)
#   • DNS konfigurálás (távolság: 0.456)
#
# 🤖 LLM válasz generálása...
#
# ======================================================================
# VÁLASZ:
# ======================================================================
# A domain beállításához először lépj be a cPanel-be, majd válaszd ki
# a "Domainek" menüpontot. Itt hozzáadhatsz új domaint az "Új domain
# hozzáadása" gombra kattintva. Add meg a domain nevet, és válaszd ki,
# hogy fődomainként vagy aldomainként szeretnéd használni.
# ======================================================================
#
# 📚 Források:
#   • https://www.rackhost.hu/tudasbazis/domain/domain-beallitas/
```

## 📁 Project Structure

```
rackhostllm/
├── data/
│   ├── kb_export.jsonl      # Raw scraped data
│   ├── kb_clean.jsonl       # Cleaned articles
│   └── kb_chunks.jsonl      # Chunked documents
├── scripts/
│   ├── scraper.py           # Web scraping logic
│   ├── build_kb_clean.py    # Data cleaning
│   ├── chunk_kb.py          # Text chunking
│   └── build_index.py       # Vector index creation
├── rag/
│   ├── rag_qa.py            # Legacy (flan-t5-small)
│   ├── rag_qa_ollama.py     # Production RAG pipeline
│   └── rag_cli.py           # Simple CLI interface
├── chroma_kb/               # Vector database (auto-generated)
└── requirements.txt
```

## 🎯 Key Implementation Details

### Chunking Strategy

The system uses a sophisticated chunking algorithm with:
- **1200 characters per chunk** (optimal for embeddings)
- **200 character overlap** (preserves context across boundaries)
- **Word-boundary detection** (avoids splitting mid-word)

```python
# Example from chunk_kb.py
MAX_CHARS = 1200
OVERLAP_CHARS = 200

def chunk_text(text, max_chars=1200, overlap=200):
    # Splits text intelligently while preserving context
    # Returns list of overlapping chunks
```

### Retrieval Logic

Semantic search with distance-based filtering:

```python
# Retrieves top 2 most relevant documents
# Filters out results with distance > 1.5 (poor matches)
# Returns context with metadata (title, URL, category)
```

### Prompt Engineering

Optimized system prompt for Hungarian customer support:

```
Te egy Rackhost ügyfélszolgálati munkatárs vagy.

FELADAT:
- Válaszolj magyarul, 2-4 mondatban
- Használd a tudásbázis dokumentumokat
- Légy pontos, de közérthető
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Query Latency** | 2-3 seconds (end-to-end) |
| **Retrieval Accuracy** | ~85% (subjective evaluation) |
| **Memory Usage** | ~2.5GB (Mistral 7B loaded) |
| **Index Size** | ~500 KB (embeddings) |
| **Document Coverage** | 100% of public KB |

## 🔧 Configuration

Key parameters can be adjusted in `rag_qa_ollama.py`:

```python
# Number of documents to retrieve
TOP_K_DOCS = 2

# Maximum context length per document
MAX_CONTEXT_CHARS = 1200

# Distance threshold for filtering results
DISTANCE_THRESHOLD = 1.5

# LLM model (swap for llama3.2:3b if Mistral is slow)
OLLAMA_MODEL = "mistral:latest"
```

## 🎓 Learning Journey

This is my first real programming project. Key milestones:

1. ✅ Started with `flan-t5-small` (failed - too weak for Hungarian)
2. ✅ Migrated to Ollama/Mistral (significant quality improvement)
3. ✅ Optimized for Mac M2 (8GB RAM constraints)
4. ✅ Implemented smart chunking with overlap
5. ✅ Added fallback mechanisms for reliability

### Challenges Overcome

- **Model Selection**: Learned that model size matters - flan-t5-small (60M params) couldn't generate coherent Hungarian
- **Memory Management**: Had to disable MPS and optimize for 8GB RAM
- **Hungarian Language**: Required custom prompting and careful model selection
- **Retrieval Quality**: Implemented distance thresholding to filter irrelevant results

## 🚧 Future Improvements

- [ ] Add evaluation metrics (RAGAS-style)
- [ ] Implement query preprocessing (spell check, expansion)
- [ ] Add caching layer for repeated queries
- [ ] Create simple web UI (Flask/Streamlit)
- [ ] Fine-tune embeddings on domain-specific data
- [ ] Implement hybrid search (keyword + semantic)
- [ ] Add user feedback mechanism

## 📝 Requirements

See `requirements.txt`:

```
chromadb
sentence-transformers
requests
python-dotenv
```

## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are welcome!

## 📄 License

MIT License - feel free to use this for learning purposes.

## 👤 Author

**Ákos Berkesi**
- Role: AI Product Owner
- Background: First programming project, self-taught
- Connect: [LinkedIn](https://www.linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Rackhost for the knowledge base content (used with permission for internal tooling)
- Anthropic Claude for coding assistance and architecture guidance
- Ollama team for making local LLMs accessible
- ChromaDB for the excellent vector database

---

⭐ **If you found this helpful, consider starring the repo!**

*Built with ❤️ and a lot of Stack Overflow searches*
