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
│   │   ├── database.py      # SQLite operations
│   │   └── vectorstore.py   # Chroma vector store
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py          # API endpoints
│   └── services/
│       ├── __init__.py
│       └── rag_service.py   # RAG chain logic
├── docs/                    # Sample CSV data (optional)
├── .env.example            # Environment template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- Google API Key (for Gemini AI) - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

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

Copy the example file and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your Google API Key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## ▶️ Running the Application

### Development Mode

```bash
python -m app.main
```

Or from the app directory:

```bash
cd app
python main.py
```

The API will start at `http://localhost:8090`

### Using Uvicorn Directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```

## 📚 API Documentation

Once the server is running, visit:

| Documentation | URL |
|-------------|-----|
| Swagger UI | http://localhost:8090/docs |
| ReDoc | http://localhost:8090/redoc |
| OpenAPI JSON | http://localhost:8090/openapi.json |

## 📡 API Endpoints

### 1. Health Check

```http
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "service": "RAG Chatbot API"
}
```

### 2. Chat with RAG Bot

```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What is the batting average of Virat Kohli in Test matches?",
  "session_id": "optional-session-id",
  "model": "gemini-2.5-flash"
}
```

Response:
```json
{
  "answer": "Virat Kohli has a Test batting average of 49.15...",
  "session_id": "uuid-here",
  "model": "gemini-2.5-flash"
}
```

### 3. Upload Document

```bash
# Using cURL
curl -X POST -F "file=@your_data.csv" http://localhost:8090/api/v1/upload-doc
```

Response:
```json
{
  "message": "CSV your_data.csv indexed successfully.",
  "file_id": 1,
  "filename": "your_data.csv"
}
```

### 4. List Documents

```http
GET /api/v1/documents
```

### 5. Delete Document

```http
DELETE /api/v1/documents/{file_id}
```

## ⚙️ Configuration

All settings can be configured via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | **Required**: Google API Key |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8090 | Server port |
| `DEBUG` | false | Debug mode |
| `DB_NAME` | rag_app.db | SQLite database name |
| `CHROMA_PERSIST_DIRECTORY` | ./chroma_db | Vector store directory |
| `CHROMA_CHUNK_SIZE` | 500 | Chunk size for text splitting |
| `CHROMA_CHUNK_OVERLAP` | 50 | Chunk overlap for text splitting |
| `EMBEDDING_MODEL` | models/gemini-embedding-001 | Embedding model |
| `DEFAULT_MODEL` | gemini-2.5-flash | Default LLM model |
| `LLM_TEMPERATURE` | 0.0 | LLM temperature (0-1) |
| `RETRIEVER_K` | 2 | Number of documents to retrieve |
| `LANGCHAIN_TRACING_V2` | true | Enable LangChain tracing |
| `LANGCHAIN_PROJECT` | rag-chatbot | LangChain project name |
| `LANGCHAIN_API_KEY` | - | LangChain API key (for LangSmith) |

## 🧪 Testing

### Using Swagger UI

1. Start the server: `python -m app.main`
2. Open http://localhost:8090/docs
3. Try the endpoints interactively

### Using cURL

```bash
# Health check
curl http://localhost:8090/api/v1/health

# Upload a document
curl -X POST -F "file=@docs/cricket_test_rag.csv" http://localhost:8090/api/v1/upload-doc

# Chat with the bot
curl -X POST http://localhost:8090/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Who has the highest batting average?", "model": "gemini-2.5-flash"}'
```

## 📝 Sample Data Format

Your CSV files should have headers that describe the data. Example:

```csv
Player_Name,Format,Country,Matches,Innings,Total_Runs,Avg,Strike_Rate,Role
V. Kohli,Test,India,113,191,8848,49.15,55.5,Middle Order
V. Kohli,ODI,India,311,299,14500,58.71,93.5,Middle Order
D.G. Bradman,Test,Australia,52,80,6996,99.94,62.4,Middle Order
```

## 🏗️ Architecture

This project follows **Clean Architecture** principles:

```
┌─────────────────────────────────────────────────────────────┐
│                      API Routes                              │
│                   (app/routes/chat.py)                       │
├─────────────────────────────────────────────────────────────┤
│                      Services                                │
│                 (app/services/rag_service.py)                │
├──────────────────────┬──────────────────────────────────────┤
│   Repositories       │           Models                      │
│  - database.py       │        (app/models/schemas.py)        │
│  - vectorstore.py    │                                       │
├──────────────────────┴──────────────────────────────────────┤
│                      Config                                   │
│                   (app/config.py)                            │
└─────────────────────────────────────────────────────────────┘
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- The `.gitignore` excludes `.env`, `chroma_db/`, and `rag_app.db`
- Use `.env.example` as a template for configuration
- Keep your API keys secure

## 🔍 LangChain Tracing (LangSmith)

To monitor your LLM calls and chain execution:

1. Set `LANGCHAIN_TRACING_V2=true` in your `.env` file
2. Add your LangChain API key: `LANGCHAIN_API_KEY=your_langchain_api_key`
3. Visit [LangSmith Dashboard](https://smith.langchain.com/) to view traces

Traces will show:
- LLM prompts and responses
- Document retrieval
- Chain execution flow
- Latency and token usage

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://langchain.com/) - LLM application framework
- [Chroma](https://www.trychroma.com/) - Vector database
- [Google Gemini](https://gemini.google.com/) - LLM provider
