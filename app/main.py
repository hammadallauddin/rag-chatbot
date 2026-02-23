"""
RAG Chatbot API - Main Application Entry Point.

A FastAPI-based REST API for a Retrieval-Augmented Generation chatbot
using LangChain, Chroma vector store, and Google Gemini AI.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.repositories.database import db_repository
from app.routes.chat import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting RAG Chatbot API...")
    
    # Debug: Log all settings
    logger.info(f"GOOGLE_API_KEY: {settings.google_api_key[:10] if settings.google_api_key else 'NOT SET'}...")
    logger.info(f"LANGCHAIN_API_KEY: {settings.langchain_api_key[:10] if settings.langchain_api_key else 'NOT SET'}...")
    logger.info(f"LANGCHAIN_TRACING_V2: {settings.langchain_tracing_v2}")
    logger.info(f"LANGCHAIN_PROJECT: {settings.langchain_project}")
    
    # Set environment variables from settings
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key or ""
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key or ""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project or "rag-chatbot"
    
    db_repository.initialize_tables()
    logger.info("Database initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## RAG Chatbot API

A powerful Retrieval-Augmented Generation (RAG) chatbot API powered by:
- **FastAPI** - Modern, fast web framework
- **LangChain** - LLM application framework
- **Chroma** - Vector database for embeddings
- **Google Gemini** - Large language model

### Features
- Chat with context-aware AI using RAG
- Upload and index CSV documents
- Session-based conversation history
- Multiple model support (Gemini 2.0 Flash, Gemini 2.5 Flash)

### Quick Start
1. Upload a CSV document using `/api/v1/upload-doc`
2. Ask questions using `/api/v1/chat`
3. Enjoy contextual answers!
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
