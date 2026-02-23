"""API routes for chat operations."""
import os
import uuid
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.models.schemas import QueryInput, QueryResponse, UploadResponse
from app.services.rag_service import rag_service, RAGServiceError
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
    
    The bot will use uploaded documents to provide context-aware answers.
    If no documents are uploaded, it will respond without context.
    """
    # Generate session ID if not provided
    session_id = query_input.session_id or str(uuid.uuid4())
    
    # Validate question
    if not query_input.question or not query_input.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    logger.info(f"Chat request - Session: {session_id}, Question: {query_input.question[:50]}..., Model: {query_input.model.value}")

    try:
        # Get chat history from database
        chat_history = db_repository.get_chat_history(session_id)
        logger.debug(f"Retrieved {len(chat_history)} chat history messages")

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

    except RAGServiceError as e:
        logger.error(f"RAG service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing chat request: {e}")
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
    
    The file will be processed and added to the vector store for RAG queries.
    """
    # Validate file extension
    allowed_extensions = ['.csv']
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided."
        )
    
    file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only CSV files are supported. Received: {file_extension}"
        )

    # Save temporary file
    temp_file_path = f"temp_{file.filename}"

    try:
        # Write uploaded file to temp location
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided."
            )
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        # Insert document record
        file_id = db_repository.insert_document_record(file.filename)
        logger.info(f"Created document record for: {file.filename} with file_id: {file_id}")

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
                detail=f"Failed to index {file.filename}. Check logs for details."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")


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
            logger.info(f"Deleted document with file_id: {file_id}")
            return {"message": f"Document {file_id} deleted successfully."}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting document {file_id}: {e}")
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
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
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
    return {
        "status": "healthy", 
        "service": "RAG Chatbot API",
        "version": "1.0.0"
    }


@router.get(
    "/debug/vectorstore-count",
    summary="Debug: Get vector store count",
    description="Get the number of documents in the vector store"
)
async def get_vectorstore_count():
    """Debug endpoint to check vector store."""
    try:
        from app.repositories.vectorstore import vectorstore_repository
        count = vectorstore_repository.vectorstore._collection.count()
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting vectorstore count: {e}")
        return {"error": str(e), "count": 0}


@router.get(
    "/debug/vectorstore-sample",
    summary="Debug: Get sample documents from vector store",
    description="Get sample documents from the vector store"
)
async def get_vectorstore_sample():
    """Debug endpoint to see what's in the vector store."""
    try:
        from app.repositories.vectorstore import vectorstore_repository
        docs = vectorstore_repository.vectorstore.get()
        return {
            "count": len(docs.get("ids", [])),
            "ids": docs.get("ids", [])[:5],
            "metadatas": docs.get("metadatas", [])[:5]
        }
    except Exception as e:
        logger.error(f"Error getting vectorstore sample: {e}")
        return {"error": str(e)}
