import os
import time
import streamlit as st
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up directories where files will be stored
DATA_DIR = "data"
VECTORDB_DIR = "vectordb"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file")
    st.error("Error: GROQ_API_KEY not found in .env file. Please add it to your .env file.")
    st.stop()

# Set API key for Groq
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Initialize the LLM using Groq
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="llama3-8b-8192",
    temperature=0
)

# Setup document processing libraries
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
)

# Text chunking for documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Use FAISS for vector storage (more stable than ChromaDB)
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Wrapper class to make the model compatible with LangChain
class SimpleEmbeddings:
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        return self.model.encode(texts)
        
    def embed_query(self, text):
        return self.model.encode(text)

# Global variables to track state
vectorstore = None
processed_documents = []

def get_loader_for_file(file_path):
    """Get the appropriate loader based on file extension"""
    file_extension = file_path.split(".")[-1].lower()
    
    if file_extension == "pdf":
        return PyPDFLoader(file_path)
    elif file_extension == "docx":
        return Docx2txtLoader(file_path)
    elif file_extension == "txt":
        return TextLoader(file_path)
    elif file_extension == "csv":
        return CSVLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)

def process_document(uploaded_file):
    """Process an uploaded document and add it to the vector database"""
    global vectorstore, processed_documents
    
    try:
        # Save the uploaded file to the data directory
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Get the appropriate loader for this file type
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            
            # Show info about the document
            st.info(f"Loaded document: {uploaded_file.name} with {len(documents)} pages/sections")
            
            # Split into chunks for processing
            chunks = text_splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks for processing")
            
            # Create embeddings for vector search
            embeddings = SimpleEmbeddings(embedding_model)
            
            # Create or update vector database
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(VECTORDB_DIR)
            else:
                vectorstore.add_documents(chunks)
                vectorstore.save_local(VECTORDB_DIR)
            
            # Record the processed document
            doc_info = {
                "filename": uploaded_file.name,
                "type": file_path.split(".")[-1],
                "size": uploaded_file.size,
                "chunks": len(chunks),
                "path": file_path,
                "timestamp": datetime.now().isoformat()
            }
            processed_documents.append(doc_info)
            
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            return doc_info
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Document processing error: {e}")
        return None

def process_web_page(url):
    """Process a web page and add it to the vector database"""
    global vectorstore, processed_documents
    
    try:
        with st.spinner(f"Processing web page: {url}..."):
            # Load the web page
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Show document info
            st.info(f"Loaded web page: {url} with {len(documents)} sections")
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks for processing")
            
            # Create embeddings
            embeddings = SimpleEmbeddings(embedding_model)
            
            # Create or update vector database
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(VECTORDB_DIR)
            else:
                vectorstore.add_documents(chunks)
                vectorstore.save_local(VECTORDB_DIR)
            
            # Record the processed web page
            doc_info = {
                "filename": url,
                "type": "web",
                "chunks": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            processed_documents.append(doc_info)
            
            st.success(f"✅ Successfully processed web page: {url}")
            return doc_info
            
    except Exception as e:
        st.error(f"Error processing web page: {str(e)}")
        logger.error(f"Web page processing error: {e}")
        return None

def process_query(query):
    """Process a user query using RAG"""
    global vectorstore
    
    start_time = time.time()
    
    try:
        if vectorstore is None:
            return "Please upload at least one document or web page first.", []
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        
        # Format sources for display
        sources = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            source_info = f"{source}"
            if page:
                source_info += f" (Page {page})"
            sources.append({
                "content": doc.page_content,
                "source": source_info
            })
        
        # Format documents as context for the LLM
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Create the prompt
        from langchain_core.prompts import ChatPromptTemplate
        template = """
        You are an AI assistant for answering questions based on the provided documents.
        
        CONTEXT INFORMATION:
        {context}
        
        QUESTION: {query}
        
        INSTRUCTIONS:
        1. Answer the question based on the context provided.
        2. If the answer isn't contained in the context, say "I don't have enough information to answer this question."
        3. Answer should be detailed and well-formatted.
        4. Do not mention that you're using context or documents in your answer.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate chain and run
        from langchain_core.output_parsers import StrOutputParser
        chain = prompt | llm | StrOutputParser()
        
        # Execute the chain
        answer = chain.invoke({"context": context, "query": query})
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Calculate a simple relevance score (could be improved)
        relevance_score = min(1.0, len(docs) / 3.0)
        
        # Record performance metrics
        performance_metrics = {
            "query": query,
            "response_time": response_time,
            "num_documents": len(docs),
            "relevance_score": relevance_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store metrics in session state
        if "performance_metrics" not in st.session_state:
            st.session_state.performance_metrics = []
        st.session_state.performance_metrics.append(performance_metrics)
        
        return answer, sources
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        st.error(f"Error processing query: {str(e)}")
        return "I encountered an error processing your query. Please try again.", []

# Try to load existing vector database if available
try:
    embeddings = SimpleEmbeddings(embedding_model)
    if os.path.exists(os.path.join(VECTORDB_DIR, "index.faiss")):
        vectorstore = FAISS.load_local(VECTORDB_DIR, embeddings)
        logger.info(f"Loaded existing vector database from {VECTORDB_DIR}")
except Exception as e:
    logger.warning(f"Could not load existing vector database: {e}")
    vectorstore = None

# Set up the Streamlit page configuration
st.set_page_config(page_title="📚 RAG Assistant", layout="wide")

# Page title and header
st.title("📚 Advanced RAG Assistant")
st.subheader("Powered by LLaMA3 on Groq")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your RAG assistant powered by LLaMA3. Upload documents or provide a web URL in the sidebar to get started."}
    ]

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = []

# Create a layout with columns
col1, col2 = st.columns([3, 1])

# Main chat area
with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                answer, sources = process_query(prompt)
                message_placeholder.markdown(answer)
                
                # Show sources if available
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:** {source['source']}")
                            st.markdown(f"```\n{source['content'][:300]}...\n```")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar for document uploads and info
with st.sidebar:
    st.header("📄 Document Processing")
    st.markdown("""
    ### Where to store documents
    
    Upload your PDF, DOCX, TXT or CSV files using the uploader below. 
    The files will be automatically saved to the `data` directory in this project.
    
    **Supported file types:**
    - PDF (`.pdf`)
    - Word documents (`.docx`)
    - Text files (`.txt`)
    - CSV files (`.csv`)
    
    After uploading, documents will be processed and stored in the vector database.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document", 
                                     type=["pdf", "docx", "txt", "csv"],
                                     help="Upload PDF, DOCX, TXT, or CSV files")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            doc_info = process_document(uploaded_file)
    
    # Web URL processing
    st.subheader("🌐 Web Page Processing")
    url_input = st.text_input("Enter a web page URL")
    
    if url_input and st.button("Process Web Page"):
        web_info = process_web_page(url_input)
    
    # Show processed documents
    if processed_documents:
        st.subheader("📚 Processed Documents")
        for i, doc in enumerate(processed_documents):
            with st.expander(f"{i+1}. {doc['filename']}"):
                st.write(f"**Type:** {doc['type']}")
                if 'chunks' in doc:
                    st.write(f"**Chunks:** {doc['chunks']}")
                if 'size' in doc:
                    st.write(f"**Size:** {doc['size']} bytes")
                st.write(f"**Added:** {doc['timestamp']}")
    else:
        st.info("No documents processed yet. Upload a document to get started.")

# Stats and metrics column
with col2:
    st.header("📊 System Stats")
    
    # Document stats
    st.metric("Documents Processed", len(processed_documents))
    
    # Query stats
    num_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Queries Processed", num_queries)
    
    # Show performance metrics if available
    if st.session_state.performance_metrics:
        st.subheader("Performance Metrics")
        
        # Calculate average response time
        avg_time = sum(m["response_time"] for m in st.session_state.performance_metrics) / len(st.session_state.performance_metrics)
        st.metric("Avg. Response Time", f"{avg_time:.2f}s")
        
        # Latest query metrics
        latest = st.session_state.performance_metrics[-1]
        st.metric("Last Query Time", f"{latest['response_time']:.2f}s")
        
        # Relevance score
        relevance = latest['relevance_score']
        st.write("Relevance Score")
        st.progress(relevance)
        st.write(f"{relevance:.2f}/1.0")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        ### Advanced RAG Assistant
        
        This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on your documents.
        
        **Features:**
        - Multi-document processing (PDF, DOCX, TXT, CSV)
        - Web page processing
        - Semantic search
        - Source citation
        
        **Technologies:**
        - LangChain
        - LLaMA3 on Groq
        - FAISS vector database
        - Sentence Transformers
        
        **File storage location:**
        - Uploaded files: `./data/`
        - Vector database: `./vectordb/`
        """)
        
        # System status
        st.markdown(f"System Status: {'🟢 Online' if GROQ_API_KEY else '🔴 API Key Missing'}")

# Tips for new users
if len(st.session_state.messages) <= 1:
    st.subheader("🔍 Getting Started")
    st.markdown("""
    1. **Upload documents** using the sidebar uploader
    2. **Process web pages** by entering URLs in the sidebar
    3. **Ask questions** about your content in the chat box below
    
    **Example questions:**
    - "What are the main topics discussed in the document?"
    - "Summarize the key points from the document."
    - "What does the document say about [specific topic]?"
    """)
