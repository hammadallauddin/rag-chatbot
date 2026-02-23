"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8090

    # Database Settings
    db_name: str = "rag_app.db"

    # Chroma Settings
    chroma_persist_directory: str = "./chroma_db"
    chroma_chunk_size: int = 500
    chroma_chunk_overlap: int = 50

    # LLM Settings
    google_api_key: str = ""
    embedding_model: str = "models/gemini-embedding-001"
    default_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # RAG Settings
    retriever_k: int = 2

    # LangChain Settings
    langchain_tracing_v2: bool = False
    langchain_project: str = "rag-chatbot"
    langchain_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
