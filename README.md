# RAG Chatbot API

A powerful Retrieval-Augmented Generation (RAG) chatbot API built with FastAPI, LangChain, Chroma vector store, and Google Gemini AI.

![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🚀 Features

- **RAG-Powered Chat**: Context-aware responses using retrieved documents
- **Document Upload**: Upload and index CSV files into vector store
- **Session Management**: Persistent conversation history per session
- **Multiple Models**: Support for Gemini 2.0 Flash and Gemini 2.5 Flash
- **RESTful API**: Built with FastAPI for high performance
- **Auto-generated Docs**: Swagger UI and ReDoc documentation
- **Clean Architecture**: Well-structured, maintainable codebase

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── database.py       # SQLite operations
│   │   └── vectorstore.py   # Chroma vector store
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py          # API endpoints
│   └── services/
│       ├── __init__.py
│       └── rag_service.py   # RAG chain logic
├── docs/                     # Sample CSV data
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- Google API Key (for Gemini AI)

### 1. Clone the Repository

```bash
git clone https://github.com/hammadallauddin/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required: Google API Key (get from https://makersuite.google.com/app/apikey)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Server Configuration
HOST=0.0.0.0
PORT=8086
DEBUG=false

# Optional: Database
DB_NAME=rag_app.db

# Optional: Chroma Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## ▶️ Running the Application

### Development Mode

```bash
cd app
python main.py
```

The API will start at `http://localhost:8086`

### Using Uvicorn Directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload
```

## 📚 API Documentation

Once the server is running, visit:

| Documentation | URL |
|-------------|-----|
| Swagger UI | http://localhost:8086/docs |
| ReDoc | http://localhost:8086/redoc |
| OpenAPI JSON | http://localhost:8086/openapi.json |

## 📡 API Endpoints

### 1. Health Check

```http
GET /api/v1/health
```

### 2. Chat with RAG Bot

```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What matches are in the dataset?",
  "session_id": "optional-session-id",
  "model": "gemini-2.5-flash"
}
```

### 3. Upload Document

```bash
curl -X POST -F "file=@matches.csv" http://localhost:8086/api/v1/upload-doc
```

### 4. List Documents

```http
GET /api/v1/documents
```

### 5. Delete Document

```http
DELETE /api/v1/documents/{file_id}
```

## 🔧 Configuration

All settings can be configured via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | Required: Google API Key |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8086 | Server port |
| `DEBUG` | false | Debug mode |
| `DB_NAME` | rag_app.db | SQLite database name |
| `CHROMA_PERSIST_DIRECTORY` | ./chroma_db | Vector store directory |
| `EMBEDDING_MODEL` | models/gemini-embedding-001 | Embedding model |
| `DEFAULT_MODEL` | gemini-2.5-flash | Default LLM model |
| `RETRIEVER_K` | 2 | Number of documents to retrieve |

## 🧪 Testing

### Using Swagger UI

1. Start the server: `python app/main.py`
2. Open http://localhost:8086/docs
3. Try the endpoints interactively

### Using cURL

```bash
# Health check
curl http://localhost:8086/api/v1/health

# Upload a document
curl -X POST -F "file=@docs/t20_matches.csv" http://localhost:8086/api/v1/upload-doc

# Chat
curl -X POST http://localhost:8086/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the dataset about?", "model": "gemini-2.5-flash"}'
```

## 📝 Sample Data

The `docs/` directory contains sample cricket match datasets:

- `test_matches.csv` - Test matches
- `one_day_matches.csv` - One Day International matches
- `t20_matches.csv` - T20 matches

## 🏗️ Architecture

This project follows **Clean Architecture** principles:

```
┌─────────────────────────────────────────────────────────────┐
│                      API Routes                              │
│                   (app/routes/chat.py)                       │
├─────────────────────────────────────────────────────────────┤
│                      Services                                │
│                 (app/services/rag_service.py)               │
├──────────────────────┬──────────────────────────────────────┤
│   Repositories       │           Models                      │
│  - database.py       │        (app/models/schemas.py)        │
│  - vectorstore.py    │                                       │
├──────────────────────┴──────────────────────────────────────┤
│                      Config                                   │
│                   (app/config.py)                            │
└─────────────────────────────────────────────────────────────┘
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [Chroma](https://www.trychroma.com/) - Vector database
- [Google Gemini](https://gemini.google.com/) - LLM provider
