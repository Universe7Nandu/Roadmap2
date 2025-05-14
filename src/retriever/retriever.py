from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    TimeWeightedVectorStoreRetriever
)
from langchain_community.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models.llms import BaseLLM
from datetime import datetime

class RetrievalManager:
    """
    A class for managing different retrieval strategies and advanced
    retrieval techniques like multi-query, contextual compression, etc.
    """
    
    @staticmethod
    def get_vector_store_retriever(
        vector_store: VectorStore,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Get a basic vector store retriever.
        
        Args:
            vector_store: Vector store to use
            search_type: Type of search ('similarity', 'mmr', 'hybrid')
            search_kwargs: Additional search arguments
        
        Returns:
            A vector store retriever
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    @staticmethod
    def get_multi_query_retriever(
        retriever: BaseRetriever,
        llm: Optional[BaseLLM] = None
    ) -> BaseRetriever:
        """
        Get a multi-query retriever that generates multiple queries
        for a single user question to improve recall.
        
        Args:
            retriever: Base retriever to use
            llm: Language model for generating multiple queries
            
        Returns:
            A multi-query retriever
        """
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
            
        try:
            # Try using the new MultiQueryRetriever with the current interface
            return MultiQueryRetriever.from_llm(
                retriever=retriever,
                llm=llm
            )
        except (TypeError, ValueError, AttributeError) as e:
            # Fallback: if the MultiQueryRetriever API has changed, revert to base retriever
            print(f"Error creating MultiQueryRetriever: {e}")
            return retriever
    
    @staticmethod
    def get_contextual_compression_retriever(
        retriever: BaseRetriever,
        llm: Optional[BaseLLM] = None
    ) -> ContextualCompressionRetriever:
        """
        Get a contextual compression retriever that filters and
        reranks retrieved documents based on relevance.
        
        Args:
            retriever: Base retriever to use
            llm: Language model for compression logic
            
        Returns:
            A contextual compression retriever
        """
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
            
        from langchain.retrievers.document_compressors import LLMChainExtractor
        
        # Create a document compressor that uses the LLM to extract relevant information
        compressor = LLMChainExtractor.from_llm(llm)
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
    
    @staticmethod
    def get_time_weighted_retriever(
        vector_store: VectorStore,
        decay_rate: float = 0.01,
        k: int = 4
    ) -> TimeWeightedVectorStoreRetriever:
        """
        Get a time-weighted retriever that considers document recency.
        
        Args:
            vector_store: Vector store to use
            decay_rate: Rate at which document relevance decays with time
            k: Number of documents to retrieve
            
        Returns:
            A time-weighted retriever
        """
        # Using the current time as a reference
        current_time = datetime.now().timestamp()
        
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vector_store,
            decay_rate=decay_rate,
            k=k,
            other_score_keys=["relevance"],
            default_time=current_time
        )
    
    @staticmethod
    def get_ensemble_retriever(retrievers: List[BaseRetriever], weights: Optional[List[float]] = None) -> Callable:
        """
        Create an ensemble retriever that combines results from multiple retrievers.
        
        Args:
            retrievers: List of retrievers to use
            weights: Optional weights for each retriever
            
        Returns:
            A function that acts as an ensemble retriever
        """
        if weights is None:
            # Equal weights if not specified
            weights = [1.0 / len(retrievers)] * len(retrievers)
            
        def retrieve_with_ensemble(query: str, **kwargs) -> List[Document]:
            all_docs = []
            unique_docs = {}
            
            for i, retriever in enumerate(retrievers):
                docs = retriever.get_relevant_documents(query, **kwargs)
                # Apply weight to each document score (assuming there's a metadata['score'])
                for doc in docs:
                    if 'score' not in doc.metadata:
                        doc.metadata['score'] = 1.0
                    doc.metadata['score'] *= weights[i]
                all_docs.extend(docs)
            
            # Remove duplicates by content and sort by weighted score
            for doc in all_docs:
                content = doc.page_content
                if content not in unique_docs or unique_docs[content].metadata.get('score', 0) < doc.metadata.get('score', 0):
                    unique_docs[content] = doc
                    
            sorted_docs = sorted(
                unique_docs.values(), 
                key=lambda x: x.metadata.get('score', 0),
                reverse=True
            )
            
            return sorted_docs
            
        return retrieve_with_ensemble
    
    @staticmethod
    def get_router_retriever(
        retrievers: Dict[str, BaseRetriever],
        llm: Optional[BaseLLM] = None
    ) -> Callable:
        """
        Create a router retriever that selects the appropriate retriever based on the query.
        
        Args:
            retrievers: Dictionary mapping retriever names to retrievers
            llm: Language model for routing logic
            
        Returns:
            A function that acts as a router retriever
        """
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
        
        def route_and_retrieve(query: str, **kwargs) -> List[Document]:
            # Prompt template for routing
            router_template = """You are a query routing system.
            Based on the user query, select the most appropriate retrieval system from the following options:
            
            {options}
            
            User query: {query}
            
            Selected retriever (just return the name of the retriever, nothing else):"""
            
            # Format options as a bulleted list
            options_text = "\n".join([f"- {name}: {description}" for name, description in {
                "semantic": "Uses semantic search for conceptual understanding",
                "keyword": "Uses keyword matching for specific terms and phrases",
                "hybrid": "Combines semantic and keyword for balanced retrieval",
                # Add other retrievers as needed
            }.items() if name in retrievers])
            
            # Format the prompt
            prompt = router_template.format(options=options_text, query=query)
            
            # Get routing decision
            response = llm.predict(prompt).strip().lower()
            
            # Default to first retriever if response doesn't match any retriever
            if response not in retrievers:
                response = list(retrievers.keys())[0]
                
            # Use the selected retriever
            selected_retriever = retrievers[response]
            return selected_retriever.get_relevant_documents(query, **kwargs)
            
        return route_and_retrieve
    
    @staticmethod
    def get_knowledge_graph_retriever(
        vector_store: VectorStore,
        llm: Optional[BaseLLM] = None,
        k: int = 4
    ) -> BaseRetriever:
        """
        Create a knowledge graph-enhanced retriever that uses entity relationships.
        This is a simplified implementation, a full KG retriever would require more setup.
        
        Args:
            vector_store: Vector store to use
            llm: Language model for entity extraction
            k: Number of documents to retrieve
            
        Returns:
            A knowledge graph-enhanced retriever
        """
        # Create a simple vector store retriever as the base
        base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        if llm is None:
            llm = ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
        
        # Entity extraction capability
        def extract_entities(query: str) -> List[str]:
            entity_prompt = """Extract the key entities from this text. Return them as a comma-separated list.
            
            Text: {query}
            
            Entities:"""
            
            entities_text = llm.predict(entity_prompt.format(query=query))
            return [e.strip() for e in entities_text.split(",")]
        
        # Enhance retrieval with entity awareness
        def retrieve_with_entities(query: str, **kwargs) -> List[Document]:
            # Get base results
            base_results = base_retriever.get_relevant_documents(query, **kwargs)
            
            # Extract entities from query
            entities = extract_entities(query)
            
            # If we have entities, enhance the search with entity-specific queries
            if entities:
                entity_results = []
                for entity in entities:
                    # Search specifically for documents mentioning this entity
                    entity_query = f"{entity} {query}"
                    entity_docs = vector_store.similarity_search(entity_query, k=2)
                    entity_results.extend(entity_docs)
                
                # Combine and deduplicate results
                all_docs = base_results + entity_results
                unique_contents = set()
                unique_docs = []
                
                for doc in all_docs:
                    if doc.page_content not in unique_contents:
                        unique_contents.add(doc.page_content)
                        unique_docs.append(doc)
                
                return unique_docs[:k]  # Return top k after enhancement
            
            return base_results
            
        # Return as a callable function
        return retrieve_with_entities
