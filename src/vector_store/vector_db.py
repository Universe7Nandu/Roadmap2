from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
import os

class VectorStoreManager:
    """
    A class for managing different vector stores (databases) used for storing
    and retrieving document embeddings.
    """
    
    @staticmethod
    def get_chroma_store(
        embedding: Embeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents"
    ) -> Chroma:
        """
        Get a Chroma vector store.
        
        Args:
            embedding: Embeddings model to use
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection
            
        Returns:
            Chroma vector store instance
        """
        return Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
    @staticmethod
    def get_faiss_store(
        embedding: Embeddings,
        index_name: str = "rag_documents"
    ) -> FAISS:
        """
        Get a FAISS vector store.
        
        Args:
            embedding: Embeddings model to use
            index_name: Name of the FAISS index
            
        Returns:
            FAISS vector store instance
        """
        # Check if the index already exists
        if os.path.exists(f"{index_name}.faiss"):
            return FAISS.load_local(
                folder_path=".",
                index_name=index_name,
                embeddings=embedding
            )
        else:
            # Return an empty FAISS object
            return FAISS(embedding_function=embedding)
    
    @staticmethod
    def get_qdrant_store(
        embedding: Embeddings,
        collection_name: str = "rag_documents",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        in_memory: bool = True
    ) -> Qdrant:
        """
        Get a Qdrant vector store.
        
        Args:
            embedding: Embeddings model to use
            collection_name: Name of the collection
            url: Qdrant server URL (if using remote server)
            api_key: Qdrant API key (if using cloud service)
            in_memory: Whether to use in-memory storage
            
        Returns:
            Qdrant vector store instance
        """
        # Use local in-memory Qdrant by default
        if in_memory:
            return Qdrant.from_params(
                embedding=embedding,
                collection_name=collection_name
            )
        else:
            # For remote Qdrant server
            if url:
                return Qdrant.from_params(
                    embedding=embedding,
                    collection_name=collection_name,
                    url=url,
                    api_key=api_key
                )
            else:
                raise ValueError("URL must be provided for non-in-memory Qdrant")
    
    @staticmethod
    def get_vector_store(
        store_type: str = "chroma",
        embedding: Optional[Embeddings] = None,
        **kwargs
    ) -> VectorStore:
        """
        Get the specified vector store.
        
        Args:
            store_type: Type of vector store ('chroma', 'faiss', 'qdrant')
            embedding: Embeddings model to use
            **kwargs: Additional arguments for the specific vector store
            
        Returns:
            A vector store instance
        """
        if embedding is None:
            raise ValueError("Embedding model must be provided")
            
        if store_type == "chroma":
            persist_directory = kwargs.get("persist_directory", "./chroma_db")
            collection_name = kwargs.get("collection_name", "rag_documents")
            return VectorStoreManager.get_chroma_store(
                embedding=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        elif store_type == "faiss":
            index_name = kwargs.get("index_name", "rag_documents")
            return VectorStoreManager.get_faiss_store(
                embedding=embedding,
                index_name=index_name
            )
        elif store_type == "qdrant":
            collection_name = kwargs.get("collection_name", "rag_documents")
            url = kwargs.get("url", None)
            api_key = kwargs.get("api_key", None)
            in_memory = kwargs.get("in_memory", True)
            return VectorStoreManager.get_qdrant_store(
                embedding=embedding,
                collection_name=collection_name,
                url=url,
                api_key=api_key,
                in_memory=in_memory
            )
        else:
            # Default to Chroma
            return VectorStoreManager.get_chroma_store(embedding=embedding)
    
    @staticmethod
    def add_documents(
        vector_store: VectorStore,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to a vector store.
        
        Args:
            vector_store: Vector store to add documents to
            documents: List of documents to add
            ids: Optional document IDs
            
        Returns:
            List of document IDs
        """
        return vector_store.add_documents(documents, ids=ids)
    
    @staticmethod
    def similarity_search(
        vector_store: VectorStore,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a similarity search on the vector store.
        
        Args:
            vector_store: Vector store to search
            query: Query string
            k: Number of results to return
            filter: Optional filters to apply
            
        Returns:
            List of relevant documents
        """
        return vector_store.similarity_search(query, k=k, filter=filter)
    
    @staticmethod
    def hybrid_search(
        vector_store: VectorStore,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a hybrid search combining vector and keyword search.
        Requires a vector store that supports hybrid search.
        
        Args:
            vector_store: Vector store to search
            query: Query string
            k: Number of results to return
            alpha: Weighting factor between vector and keyword search
            filter: Optional filters to apply
            
        Returns:
            List of relevant documents
        """
        # Check if vector store supports hybrid search
        # Different vector stores might implement hybrid search with different method names
        if hasattr(vector_store, "hybrid_search"):
            return vector_store.hybrid_search(
                query=query,
                k=k, 
                alpha=alpha,
                filter=filter
            )
        elif hasattr(vector_store, "similarity_search_with_relevance_scores"):
            # Some vector stores provide a relevance score which can be used as a form of hybrid search
            search_results = vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter
            )
            # Return just the documents
            return [doc for doc, score in search_results]
        elif hasattr(vector_store, "max_marginal_relevance_search"):
            # MMR search can be used as another alternative for hybrid-like search
            return vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=k*3,  # Fetch more candidates for reranking
                filter=filter
            )
        # Fallback to regular similarity search
        return vector_store.similarity_search(query, k=k, filter=filter)
