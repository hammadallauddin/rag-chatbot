"""RAG (Retrieval-Augmented Generation) service for chat operations."""
import logging
import os
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.repositories.vectorstore import vectorstore_repository

logger = logging.getLogger(__name__)

# LangChain project name for tracing
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "rag-chatbot")

# QA prompt for answering questions
QA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the following context to answer the user's question accurately. "
    "If you don't know the answer based on the context, say so honestly. "
    "Context: {context}"
)


class RAGServiceError(Exception):
    """Custom exception for RAG service errors."""
    pass


class RAGService:
    """Service for RAG-based chat operations."""

    def __init__(self):
        """Initialize the RAG service."""
        self._llm_cache = {}

    def _get_llm(self, model_name: str) -> ChatGoogleGenerativeAI:
        """Get or create an LLM instance (cached)."""
        if model_name not in self._llm_cache:
            # Get API key from environment directly
            api_key = os.environ.get("GOOGLE_API_KEY") or settings.google_api_key
            
            if not api_key:
                # Log what's available for debugging
                logger.error(f"GOOGLE_API_KEY env var: {os.environ.get('GOOGLE_API_KEY')}")
                logger.error(f"settings.google_api_key: {settings.google_api_key}")
                raise RAGServiceError("Google API key is not configured. Please set GOOGLE_API_KEY in .env file.")
            
            self._llm_cache[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=settings.llm_temperature,
                convert_system_message_to_human=True,
                google_api_key=api_key
            )
            logger.info(f"Created new LLM instance for model: {model_name}")
        return self._llm_cache[model_name]

    def clear_llm_cache(self) -> None:
        """Clear the LLM cache to force recreation of instances."""
        self._llm_cache.clear()
        logger.info("LLM cache cleared")

    def _convert_chat_history(self, history: List[dict]) -> List:
        """Convert database history format to LangChain message format."""
        messages = []
        if not history:
            return messages
            
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        return messages

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a string."""
        if not docs:
            return "No relevant information found in the uploaded documents."
        
        formatted_docs = []
        for doc in docs:
            # Build a readable string from metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            parts = []
            for key, value in metadata.items():
                if value is not None:
                    parts.append(f"{key}: {value}")
            
            metadata_str = ", ".join(parts) if parts else "No metadata"
            formatted_docs.append(f"[{metadata_str}]\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)

    def get_response(
        self, 
        question: str, 
        chat_history: List[dict], 
        model_name: Optional[str] = None
    ) -> str:
        """Get a response from the RAG chain.
        
        Args:
            question: The user's question
            chat_history: List of previous chat messages
            model_name: Optional model name override
            
        Returns:
            The AI's response string
            
        Raises:
            RAGServiceError: If there's an error in the RAG pipeline
        """
        if not question or not question.strip():
            raise RAGServiceError("Question cannot be empty.")
        
        model = model_name or settings.default_model
        logger.info(f"Processing chat request with model: {model}")
        
        try:
            llm = self._get_llm(model)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RAGServiceError(f"Failed to initialize LLM: {str(e)}")

        # Get retriever
        try:
            retriever = vectorstore_repository.get_retriever()
        except Exception as e:
            logger.error(f"Failed to get retriever: {e}")
            raise RAGServiceError(f"Failed to get retriever: {str(e)}")

        # Create QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}")
        ])

        # Convert chat history to LangChain format
        lc_chat_history = self._convert_chat_history(chat_history)

        # Use the original question for retrieval
        search_question = question.strip()

        # Retrieve documents
        try:
            retrieved_docs = retriever.invoke(search_question)
            logger.info(f"Retrieved {len(retrieved_docs)} documents for question: {search_question}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            retrieved_docs = []

        # Format context
        formatted_context = self._format_docs(retrieved_docs)
        
        if settings.debug:
            logger.debug(f"Formatted context: {formatted_context[:500]}...")

        # Generate answer using the QA prompt
        try:
            qa_chain = qa_prompt | llm | StrOutputParser()
            
            # Build the input for the chain
            chain_input = {
                "input": search_question,
                "context": formatted_context
            }
            
            # Only add chat_history if it exists
            if lc_chat_history:
                chain_input["chat_history"] = lc_chat_history
            
            # Configure LangChain tracing
            config = RunnableConfig(
                configurable={"project_name": LANGCHAIN_PROJECT},
                tags=["rag-chatbot", f"model-{model}"]
            )
            
            answer = qa_chain.invoke(chain_input, config=config)
            logger.info(f"Successfully generated response ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise RAGServiceError(f"Error generating response: {str(e)}")


# Singleton instance
rag_service = RAGService()
