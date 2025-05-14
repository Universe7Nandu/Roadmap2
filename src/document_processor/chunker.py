from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)

class DocumentChunker:
    """
    A class for chunking documents into smaller pieces for more effective processing
    and retrieval in a RAG system.
    """
    
    @staticmethod
    def get_recursive_chunker(
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] = None
    ) -> RecursiveCharacterTextSplitter:
        """
        Create a recursive character text splitter with the specified parameters.
        This is good for most general purpose chunking.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
            
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    @staticmethod
    def get_markdown_chunker(
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> MarkdownTextSplitter:
        """
        Create a Markdown-aware text splitter.
        This is optimal for Markdown documents.
        """
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @staticmethod
    def get_token_chunker(
        chunk_size: int = 500, 
        chunk_overlap: int = 50,
        model_name: str = "gpt-3.5-turbo"
    ) -> TokenTextSplitter:
        """
        Create a token-based text splitter.
        This is useful when you want to ensure chunks fit within LLM context windows.
        """
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )
    
    @staticmethod
    def chunk_documents(
        documents: List[Document], 
        chunker_type: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> List[Document]:
        """
        Split documents into chunks using the specified chunker.
        
        Args:
            documents: List of documents to chunk
            chunker_type: Type of chunker to use ('recursive', 'markdown', 'token')
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            A list of document chunks
        """
        if chunker_type == "recursive":
            chunker = DocumentChunker.get_recursive_chunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=kwargs.get("separators", None)
            )
        elif chunker_type == "markdown":
            chunker = DocumentChunker.get_markdown_chunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        elif chunker_type == "token":
            chunker = DocumentChunker.get_token_chunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                model_name=kwargs.get("model_name", "gpt-3.5-turbo")
            )
        else:
            # Default to recursive
            chunker = DocumentChunker.get_recursive_chunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
        return chunker.split_documents(documents)
    
    @staticmethod
    def create_semantic_chunks(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Create semantically meaningful chunks by trying to keep related content together.
        This is a more advanced version that attempts to respect semantic boundaries.
        """
        # Use recursive splitter with custom separators to respect semantic boundaries
        semantic_separators = [
            # Headers in markdown and documentation
            "## ", "### ", "#### ", "##### ", "###### ",
            # Paragraph breaks
            "\n\n", 
            # Sentence endings
            ". ", "! ", "? ",
            # Last resort fallbacks
            "\n", " ", ""
        ]
        
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=semantic_separators
        )
        
        return chunker.split_documents(documents)
