import os
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    WebBaseLoader,
    TextLoader
)
from langchain_core.documents import Document

class DocumentLoader:
    """
    A class for loading various document types and extracting their content.
    Supports PDF, DOCX, XLSX, TXT, and web pages.
    """
    
    @staticmethod
    def get_loader_for_file(file_path: str):
        """Get the appropriate loader based on file extension"""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.pdf':
            return UnstructuredPDFLoader(file_path)
        elif file_extension == '.docx':
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path)
        else:
            # Default to Unstructured for other file types
            return UnstructuredFileLoader(file_path)
            
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """Load a document from a file path"""
        loader = DocumentLoader.get_loader_for_file(file_path)
        return loader.load()
    
    @staticmethod
    def load_web_document(url: str) -> List[Document]:
        """Load document from a web URL"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    @staticmethod
    def load_documents_from_directory(directory_path: str) -> Dict[str, List[Document]]:
        """Load all documents from a directory"""
        documents = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    documents[filename] = DocumentLoader.load_document(file_path)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return documents
