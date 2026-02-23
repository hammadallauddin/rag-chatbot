"""RAG (Retrieval-Augmented Generation) service for chat operations."""
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.repositories.vectorstore import vectorstore_repository

logger = logging.getLogger(__name__)

# Contextualize prompt for handling chat history
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# QA prompt for answering questions
QA_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the following context to answer the user's question. "
    "If you don't know the answer, say so honestly. "
    "Context: {context}"
)


class RAGService:
    """Service for RAG-based chat operations."""

    def __init__(self):
        """Initialize the RAG service."""
        self._llm_cache = {}

    def _get_llm(self, model_name: str) -> ChatGoogleGenerativeAI:
        """Get or create an LLM instance (cached)."""
        if model_name not in self._llm_cache:
            self._llm_cache[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=settings.llm_temperature,
                convert_system_message_to_human=True
            )
        return self._llm_cache[model_name]

    def _convert_chat_history(self, history: List[dict]) -> List:
        """Convert database history format to LangChain message format."""
        messages = []
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
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, question: str, chat_history: List[dict], model_name: str = None) -> str:
        """Get a response from the RAG chain."""
        model = model_name or settings.default_model
        llm = self._get_llm(model)

        # Get retriever
        retriever = vectorstore_repository.get_retriever()

        # Create contextualize prompt for handling history
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Create QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Convert chat history to LangChain format
        lc_chat_history = self._convert_chat_history(chat_history)

        # If there's chat history, reformulate the question first
        if lc_chat_history:
            # Create a chain to contextualize the question
            contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
            
            # Get reformulated question
            reformulated_question = contextualize_chain.invoke({
                "input": question,
                "chat_history": lc_chat_history
            })
        else:
            reformulated_question = question

        # Retrieve documents
        retrieved_docs = retriever.invoke(reformulated_question)
        formatted_context = self._format_docs(retrieved_docs)

        # Generate answer using the QA prompt
        qa_chain = qa_prompt | llm | StrOutputParser()
        
        answer = qa_chain.invoke({
            "input": reformulated_question,
            "context": formatted_context,
            "chat_history": lc_chat_history
        })

        return answer


# Singleton instance
rag_service = RAGService()
