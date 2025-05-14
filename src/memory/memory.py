from typing import List, Dict, Any, Optional, Union
from langchain_community.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain_community.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.language_models.llms import BaseLLM
from langchain_core.vectorstores import VectorStore
from langchain_community.chat_models import ChatOpenAI
import uuid
import os
import json
from datetime import datetime

class MemoryManager:
    """
    A class for managing different types of memory systems for conversational context.
    Provides various memory implementations for maintaining conversation history.
    """
    
    @staticmethod
    def get_buffer_memory(
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None
    ) -> ConversationBufferMemory:
        """
        Get a simple buffer memory that stores all messages.
        
        Args:
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            output_key: Key for outputs (optional)
            input_key: Key for inputs (optional)
            
        Returns:
            ConversationBufferMemory instance
        """
        return ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key
        )
    
    @staticmethod
    def get_window_memory(
        k: int = 5,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None
    ) -> ConversationBufferWindowMemory:
        """
        Get a window memory that stores the last k messages.
        
        Args:
            k: Number of messages to keep in memory
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            output_key: Key for outputs (optional)
            input_key: Key for inputs (optional)
            
        Returns:
            ConversationBufferWindowMemory instance
        """
        return ConversationBufferWindowMemory(
            k=k,
            memory_key=memory_key,
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key
        )
    
    @staticmethod
    def get_summary_memory(
        llm: Optional[BaseLLM] = None,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None
    ) -> ConversationSummaryMemory:
        """
        Get a summary memory that summarizes the conversation.
        
        Args:
            llm: Language model for generating summaries
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            output_key: Key for outputs (optional)
            input_key: Key for inputs (optional)
            
        Returns:
            ConversationSummaryMemory instance
        """
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
            
        return ConversationSummaryMemory(
            llm=llm,
            memory_key=memory_key,
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key
        )
    
    @staticmethod
    def get_summary_buffer_memory(
        llm: Optional[BaseLLM] = None,
        max_token_limit: int = 2000,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None
    ) -> ConversationSummaryBufferMemory:
        """
        Get a summary buffer memory that combines buffer and summary approaches.
        
        Args:
            llm: Language model for generating summaries
            max_token_limit: Maximum number of tokens to store
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            output_key: Key for outputs (optional)
            input_key: Key for inputs (optional)
            
        Returns:
            ConversationSummaryBufferMemory instance
        """
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
            
        return ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            memory_key=memory_key,
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key
        )
    
    @staticmethod
    def get_vectorstore_memory(
        retriever,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        input_key: Optional[str] = None
    ) -> VectorStoreRetrieverMemory:
        """
        Get a vector store memory that stores conversations in a vector store.
        
        Args:
            retriever: Retriever to use for memory
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            input_key: Key for inputs (optional)
            
        Returns:
            VectorStoreRetrieverMemory instance
        """
        return VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key=memory_key,
            return_messages=return_messages,
            input_key=input_key
        )
    
    @staticmethod
    def get_redis_backed_memory(
        session_id: Optional[str] = None,
        url: str = "redis://localhost:6379",
        ttl: Optional[int] = None,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None
    ) -> ConversationBufferMemory:
        """
        Get a Redis-backed conversation memory.
        
        Args:
            session_id: Unique session identifier
            url: Redis connection URL
            ttl: Time-to-live for messages in seconds
            memory_key: Key to store memory under
            return_messages: Whether to return messages (True) or a string
            output_key: Key for outputs (optional)
            input_key: Key for inputs (optional)
            
        Returns:
            ConversationBufferMemory with Redis backend
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        message_history = RedisChatMessageHistory(
            session_id=session_id,
            url=url,
            ttl=ttl
        )
        
        return ConversationBufferMemory(
            memory_key=memory_key,
            chat_memory=message_history,
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key
        )
    
    @staticmethod
    def get_file_backed_memory(
        file_path: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> BaseMemory:
        """
        Get a file-backed memory system that persists conversations.
        
        Args:
            file_path: Path to save conversation history
            session_id: Unique session identifier
            
        Returns:
            A memory instance with file persistence
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        if file_path is None:
            file_path = f"./memory/session_{session_id}.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        # Create a simple wrapper around ConversationBufferMemory with file persistence
        memory = ConversationBufferMemory(return_messages=True)
        
        # Load existing conversation if file exists
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    saved_data = json.load(f)
                    for entry in saved_data:
                        memory.chat_memory.add_user_message(entry['user'])
                        memory.chat_memory.add_ai_message(entry['ai'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading memory file: {e}")
        
        # Add methods for saving to file
        def save_context(inputs, outputs):
            memory.chat_memory.add_user_message(inputs['input'])
            memory.chat_memory.add_ai_message(outputs['output'])
            
            # Save to file
            history = []
            messages = memory.chat_memory.messages
            for i in range(0, len(messages), 2):
                if i+1 < len(messages):
                    history.append({
                        'user': messages[i].content,
                        'ai': messages[i+1].content,
                        'timestamp': datetime.now().isoformat()
                    })
            
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2)
                
        memory.save_context = save_context
                
        return memory
