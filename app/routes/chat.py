"""API routes for chat operations."""
import uuid
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from app.models.schemas import QueryInput, QueryResponse, UploadResponse, DeleteFileRequest
from app.services.rag_service import rag_service
from app.repositories.database import db_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post(
    "/chat",
    response_model=QueryResponse,
    summary="Chat with RAG bot",
    description="Ask a question and get an answer from the RAG-powered chatbot"
)
async def chat(query_input: QueryInput):
    """
    Chat endpoint for RAG-based question answering.

    - **question**: The user's question
    - **session_id**: Optional session ID for conversation continuity
    - **model**: The LLM model to use (gemini-2.0-flash or gemini-2.5-flash)
    """
    # Generate session ID if not provided
    session_id = query_input.session_id or str(uuid.uuid4())

    logger.info(f"Chat request - Session: {session_id}, Question: {query_input.question}, Model: {query_input.model.value}")

    try:
        # Get chat history from database
        chat_history = db_repository.get_chat_history(session_id)

        # Get response from RAG service
        answer = rag_service.get_response(
            question=query_input.question,
            chat_history=chat_history,
            model_name=query_input.model.value
        )

        # Save chat log to database
        db_repository.insert_chat_log(
            session_id=session_id,
            user_query=query_input.question,
            gpt_response=answer,
            model=query_input.model.value
        )

        return QueryResponse(
            answer=answer,
            session_id=session_id,
            model=query_input.model
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@router.post(
    "/upload-doc",
    response_model=UploadResponse,
    summary="Upload and index document",
    description="Upload a CSV file to be indexed into the vector store"
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a CSV document.

    - **file**: CSV file to upload and index
    """
    # Validate file extension
    allowed_extensions = ['.csv']
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''

    if f'.{file_extension}' not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported."
        )

    # Save temporary file
    temp_file_path = f"temp_{file.filename}"

    try:
        # Write uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Insert document record
        file_id = db_repository.insert_document_record(file.filename)

        # Index to vector store
        from app.repositories.vectorstore import vectorstore_repository
        success = vectorstore_repository.index_document(temp_file_path, file_id)

        if success:
            logger.info(f"Successfully indexed document: {file.filename} with file_id: {file_id}")
            return UploadResponse(
                message=f"CSV {file.filename} indexed successfully.",
                file_id=file_id,
                filename=file.filename
            )
        else:
            # Rollback database record
            db_repository.delete_document_record(file_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to index {file.filename}."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )
    finally:
        # Clean up temp file
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.delete(
    "/documents/{file_id}",
    summary="Delete a document",
    description="Delete a document from the vector store and database"
)
async def delete_document(file_id: int):
    """Delete a document by its ID."""
    try:
        # Delete from vector store
        from app.repositories.vectorstore import vectorstore_repository
        vector_deleted = vectorstore_repository.delete_document(file_id)

        # Delete from database
        db_deleted = db_repository.delete_document_record(file_id)

        if vector_deleted or db_deleted:
            return {"message": f"Document {file_id} deleted successfully."}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get(
    "/documents",
    summary="List all documents",
    description="Get a list of all indexed documents"
)
async def list_documents():
    """Get all document records."""
    try:
        documents = db_repository.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running"
)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Chatbot API"}
