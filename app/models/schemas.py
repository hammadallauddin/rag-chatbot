"""Pydantic models for request/response validation."""
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class ModelName(str, Enum):
    """Available LLM models."""
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


class QueryInput(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., description="User's question")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    model: ModelName = Field(default=ModelName.GEMINI_2_5_FLASH, description="LLM model to use")


class QueryResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="AI's response")
    session_id: str = Field(..., description="Session ID")
    model: ModelName = Field(..., description="Model used for generation")


class DocumentInfo(BaseModel):
    """Document metadata model."""
    id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")


class DeleteFileRequest(BaseModel):
    """Request model for deleting a document."""
    file_id: int = Field(..., description="ID of the file to delete")


class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str = Field(..., description="Success message")
    file_id: int = Field(..., description="Uploaded file ID")
    filename: str = Field(..., description="Uploaded filename")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error details")
