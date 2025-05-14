from typing import List, Dict, Any, Optional, Union
from langchain_community.embeddings import (
    HuggingFaceEmbeddings, 
    OpenAIEmbeddings,
    MistralAIEmbeddings
)
from langchain.embeddings.base import Embeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.cache import CacheBackedEmbeddings
import os

class EmbeddingManager:
    """
    A class for managing different embedding models and strategies.
    Provides support for local (HuggingFace) and API-based embedding models.
    """
    
    @staticmethod
    def get_huggingface_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
        """
        Get Hugging Face embeddings model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            
        Returns:
            HuggingFaceEmbeddings instance
        """
        # Use an updated initialization method compatible with the latest HuggingFaceEmbeddings
        try:
            # First attempt: Modern method
            return HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
        except (TypeError, ValueError):
            try:
                # Second attempt: Without normalize_embeddings
                return HuggingFaceEmbeddings(model_name=model_name)
            except (TypeError, ValueError):
                # Fallback: Import model directly
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                
                # Create a custom embeddings class
                class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        return self.model.encode(texts)
                    
                    def embed_query(self, text):
                        return self.model.encode(text)
                
                return CustomHuggingFaceEmbeddings(model)
    
    @staticmethod
    def get_openai_embeddings(model_name: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
        """
        Get OpenAI embeddings model.
        
        Args:
            model_name: Name of the OpenAI embedding model
            
        Returns:
            OpenAIEmbeddings instance
        """
        # Note: This will use the GROQ_API_KEY from the environment with the OPENAI_API_BASE set to Groq
        return OpenAIEmbeddings(model=model_name)
    
    @staticmethod
    def get_mistral_embeddings(model_name: str = "mistral-embed") -> MistralAIEmbeddings:
        """
        Get Mistral AI embeddings model.
        
        Args:
            model_name: Name of the Mistral AI embedding model
            
        Returns:
            MistralAIEmbeddings instance
        """
        # This would require an actual Mistral API key
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            return MistralAIEmbeddings(model=model_name)
        else:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    @staticmethod
    def get_cached_embeddings(
        underlying_embeddings: Any,
        cache_dir: str = "./embedding_cache"
    ) -> CacheBackedEmbeddings:
        """
        Create a cached version of embeddings to reduce API calls and costs.
        
        Args:
            underlying_embeddings: The base embeddings to cache
            cache_dir: Directory to store cached embeddings
            
        Returns:
            CacheBackedEmbeddings instance
        """
        store = LocalFileStore(cache_dir)
        
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            store,
            namespace=underlying_embeddings.__class__.__name__
        )
    
    @staticmethod
    def get_embeddings(
        embedding_type: str = "huggingface",
        model_name: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings, MistralAIEmbeddings, CacheBackedEmbeddings]:
        """
        Get the specified embedding model.
        
        Args:
            embedding_type: Type of embeddings ('huggingface', 'openai', 'mistral')
            model_name: Name of the model to use (defaults to a sensible default per type)
            use_cache: Whether to cache embeddings to reduce API calls
            
        Returns:
            An embeddings instance
        """
        # Set default model names if not specified
        if model_name is None:
            if embedding_type == "huggingface":
                model_name = "all-MiniLM-L6-v2"
            elif embedding_type == "openai":
                model_name = "text-embedding-ada-002"
            elif embedding_type == "mistral":
                model_name = "mistral-embed"
        
        # Get the appropriate embeddings
        if embedding_type == "huggingface":
            embeddings = EmbeddingManager.get_huggingface_embeddings(model_name)
        elif embedding_type == "openai":
            embeddings = EmbeddingManager.get_openai_embeddings(model_name)
        elif embedding_type == "mistral":
            embeddings = EmbeddingManager.get_mistral_embeddings(model_name)
        else:
            # Default to Hugging Face
            embeddings = EmbeddingManager.get_huggingface_embeddings(model_name)
            
        # Apply caching if requested
        if use_cache:
            cache_dir = kwargs.get("cache_dir", "./embedding_cache")
            return EmbeddingManager.get_cached_embeddings(embeddings, cache_dir)
            
        return embeddings
