"""Vector store repository for managing Chroma operations."""
from typing import List
import logging

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreRepository:
    """Repository for vector store operations."""

    def __init__(self):
        """Initialize."""
        self._text_splitter = None
        self._embedding_function = None
        self._vectorstore = None

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy initialization of text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chroma_chunk_size,
                chunk_overlap=settings.chroma_chunk_overlap,
                length_function=len
            )
        return self._text_splitter

    @property
    def embedding_function(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy initialization of embedding function."""
        if self._embedding_function is None:
            self._embedding_function = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                task_type="retrieval_document"
            )
        return self._embedding_function

    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization of vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embedding_function
            )
        return self._vectorstore

    def load_and_split_document(self, file_path: str) -> List[Document]:
        """Load and split a CSV document into chunks."""
        if not file_path.endswith('.csv'):
            raise ValueError(f"Only CSV files are supported. Received: {file_path}")

        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def index_document(self, file_path: str, file_id: int) -> bool:
        """Index a document to the vector store."""
        try:
            splits = self.load_and_split_document(file_path)

            for split in splits:
                split.metadata['file_id'] = file_id

            self.vectorstore.add_documents(splits)
            logger.info(f"Successfully indexed document with file_id: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False

    def delete_document(self, file_id: int) -> bool:
        """Delete a document from the vector store."""
        try:
            existing_data = self.vectorstore.get(where={"file_id": file_id})
            ids_to_delete = existing_data['ids']

            if not ids_to_delete:
                logger.warning(f"No documents found for file_id {file_id}")
                return False

            self.vectorstore.delete(ids=ids_to_delete)
            logger.info(f"Successfully deleted {len(ids_to_delete)} chunks for file_id {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document with file_id {file_id}: {e}")
            return False

    def get_retriever(self):
        """Get a retriever instance from the vector store."""
        return self.vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})


# Singleton instance
vectorstore_repository = VectorStoreRepository()
