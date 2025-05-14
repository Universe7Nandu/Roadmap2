import os
import time
import streamlit as st
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up directories where we'll store files
DATA_DIR = "data"
VECTORDB_DIR = "vectordb"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

# Load environment variables (.env file)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file")
    st.error("Error: GROQ_API_KEY not found in .env file. Please add it to your .env file.")
    st.stop()

# Set API key for Groq
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Initialize the LLaMA3 model through Groq
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="llama3-8b-8192",
    temperature=0
)

# Set up document processing tools
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
)

# Set up text splitter for chunking documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Use FAISS for vector storage - more reliable than Chroma in some environments
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Custom embeddings class that works with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        return self.model.encode(texts)
        
    def embed_query(self, text):
        return self.model.encode(text)

# Global variables to store document information and vector database
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
    """Process an uploaded document and store it in the vector database."""
    global vectorstore, processed_documents
    
    try:
        # Create a unique filename
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Load the document with the appropriate loader
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            
            # Show document info
            st.info(f"Loaded document: {uploaded_file.name} with {len(documents)} pages/sections")
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks for processing")
            
            # Create embeddings instance for LangChain
            embeddings = SentenceTransformerEmbeddings(model=embedding_model)
            
            # Create or update the vector database
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                # Save the index
                vectorstore.save_local(VECTORDB_DIR)
            else:
                # Add the new document to the existing vectorstore
                vectorstore.add_documents(chunks)
                # Save the updated index
                vectorstore.save_local(VECTORDB_DIR)
            
            # Add to processed documents list
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

# Try to load existing vector database if available
try:
    embeddings = SentenceTransformerEmbeddings(model=embedding_model)
    if os.path.exists(os.path.join(VECTORDB_DIR, "index.faiss")):
        vectorstore = FAISS.load_local(VECTORDB_DIR, embeddings)
        logger.info(f"Loaded existing vector database from {VECTORDB_DIR}")
except Exception as e:
    logger.warning(f"Could not load existing vector database: {e}")
    vectorstore = None

def process_web_page(url):
    """Process a web page and add it to the vector database"""
    global vectorstore, processed_documents
    
    try:
        with st.spinner(f"Processing web page: {url}..."):
            # Load the web page content
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Show document info
            st.info(f"Loaded web page: {url} with {len(documents)} sections")
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            st.info(f"Split into {len(chunks)} chunks for processing")
            
            # Create embeddings instance for LangChain
            embeddings = SentenceTransformerEmbeddings(model=embedding_model)
            
            # Create or update the vector database
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                # Save the index
                vectorstore.save_local(VECTORDB_DIR)
            else:
                # Add the new document to the existing vectorstore
                vectorstore.add_documents(chunks)
                # Save the updated index
                vectorstore.save_local(VECTORDB_DIR)
            
            # Add to processed documents list
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
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show processing message
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Load the document
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            
            # Create chunks
            chunks = text_splitter.split_documents(documents)
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{
                'source': file_path,
                'page': chunk.metadata.get('page', None) if hasattr(chunk, 'metadata') else None,
                'chunk_id': i
            } for i, chunk in enumerate(chunks)]
            
            # Add to ChromaDB collection
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            # Create file info for display
            file_info = {
                "name": uploaded_file.name,
                "type": uploaded_file.name.split(".")[-1],
                "size": uploaded_file.size,
                "chunks": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save file info to session state
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = []
            st.session_state.processed_files.append(file_info)
            
            # Display success
            st.success(f"Document processed: {uploaded_file.name} ({len(chunks)} chunks created)")
            
        return file_info
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"Error processing document: {str(e)}")
        return None

def process_url(url):
    """Process a web page URL and store it in the vector database."""
    try:
        # Show loading animation
        with st.spinner(f"Processing web page {url}..."):
            # Load the web page
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Create chunks
            chunks = text_splitter.split_documents(documents)
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{
                'source': url,
                'chunk_id': i
            } for i, chunk in enumerate(chunks)]
            
            # Add to ChromaDB collection
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            # Create file info for display
            file_info = {
                "name": url,
                "type": "web",
                "size": sum(len(doc.page_content) for doc in documents),
                "chunks": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save file info to session state
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = []
            st.session_state.processed_files.append(file_info)
            
            # Display success
            st.success(f"Web page processed: {url} ({len(chunks)} chunks created)")
            
        return file_info
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        st.error(f"Error processing URL: {str(e)}")
        return None

def process_query(query):
    """Process a user query and generate a response using the RAG system."""
    # Check if we have documents in the collection
    if collection.count() == 0:
        st.warning("Please upload a document first.")
        return
    
    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    try:
        # Show thinking indicator
        with st.spinner("Thinking..."):
            # Start timer
            start_time = time.time()
            
            # Query ChromaDB for relevant documents
            results = collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Extract the retrieved documents
            documents = results["documents"][0]  # First query result's documents
            metadatas = results["metadatas"][0]  # First query result's metadatas
            
            # Combine documents into context
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
            
            # Create prompt for the LLM
            prompt = f"""You are a helpful assistant that answers questions based on the provided context and knowledge.
            
            Context:
            {context}
            
            User Question: {query}
            
            Please provide a comprehensive answer based on the context provided. If the answer is not in the context, say so politely and provide the best response you can based on your general knowledge, but make it clear that this information is not from the provided documents.
            
            Answer:"""
            
            # Get answer from LLM
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare source documents for display
            sources = []
            for i, metadata in enumerate(metadatas[:3]):  # Show top 3 sources
                sources.append({
                    "title": metadata.get("source", f"Document {i+1}"),
                    "page": metadata.get("page", None)
                })
            
            # Create a formatted answer with sources
            formatted_answer = answer
            if sources:
                formatted_answer += "\n\nSources:\n"
                for i, source in enumerate(sources):
                    page_info = f" (Page {source['page']})" if source.get('page') else ""
                    formatted_answer += f"- {source['title']}{page_info}\n"
            
            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            
            # Simple evaluation - calculate relevance score based on similarity
            similarity_scores = results["distances"][0] if "distances" in results else []
            relevance_score = 1.0 - (sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0.0
            
            # Save performance metrics
            performance_metrics = {
                "response_time": response_time,
                "num_documents": len(documents),
                "relevance_score": relevance_score,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store metrics in session state
            if "performance_metrics" not in st.session_state:
                st.session_state.performance_metrics = []
            st.session_state.performance_metrics.append(performance_metrics)
            
            # Debug output
            logger.info(f"Query processed in {response_time:.2f} seconds")
            logger.info(f"Retrieved {len(documents)} documents")
            logger.info(f"Relevance score: {relevance_score:.2f}")
            
            return answer, sources
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        st.error(f"Error processing query: {str(e)}")
        return None, None

# Set up the Streamlit configuration once at the beginning of the script
if 'page_config_done' not in st.session_state:
    st.set_page_config(page_title="📚 GenAI RAG Assistant", layout="wide")
    st.session_state.page_config_done = True

# Page header
st.title("📚 Advanced RAG Assistant")
st.subheader("Powered by LLaMA3 on Groq")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to the Advanced RAG Assistant! Upload a document or provide a URL to get started, then ask me any questions about your documents."}
    ]

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = []

# Main layout with columns
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        page_info = f" (Page {source['page']})" if source.get('page') else ""
                        st.write(f"- {source['title']}{page_info}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the query and get response
        answer, sources = process_query(prompt)
        
        # If we got a valid response, it's already been added to the session state
        # Just scroll to the bottom to show the new messages

with col2:
    # Sidebar for document processing
    st.sidebar.title("Document Processing")
    st.sidebar.write("Upload documents or provide URLs to build your knowledge base.")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a Document", 
        type=["pdf", "docx", "txt", "xlsx", "csv"]
    )
    
    if uploaded_file:
        if st.sidebar.button("Process Document"):
            process_document(uploaded_file)
    
    # URL input
    st.sidebar.subheader("Process Web Page")
    url_input = st.sidebar.text_input("Enter URL:", placeholder="https://example.com")
    
    if url_input and st.sidebar.button("Process URL"):
        process_url(url_input)
    
    # Document statistics
    st.subheader("System Statistics")
    
    # Documents processed
    st.metric("Documents Processed", len(st.session_state.processed_files))
    
    # Vectors in database
    try:
        st.metric("Vectors in Database", collection.count())
    except:
        st.metric("Vectors in Database", "0")
    
    # Queries processed
    num_queries = len([m for m in st.session_state.messages if m.get("role") == "user"])
    st.metric("Queries Processed", num_queries)
    
    # Performance metrics
    if st.session_state.performance_metrics:
        latest_metrics = st.session_state.performance_metrics[-1]
        
        st.subheader("Performance Metrics")
        
        # Response time
        st.metric("Response Time", f"{latest_metrics.get('response_time', 0):.2f}s")
        
        # Relevance score
        relevance = latest_metrics.get('relevance_score', 0)
        st.write("Relevance Score")
        st.progress(relevance)
        st.write(f"{relevance:.2f}/1.0")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings", expanded=False):
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Semantic", "Hybrid"],
            index=0
        )
        
        # Chunk settings
        chunk_size = st.number_input("Chunk Size", 100, 2000, 500, 50)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 100, 10)
        
        # LLM temperature
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # About section
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        ### Advanced RAG Assistant
        
        This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on your documents.
        
        **Features:**
        - Multi-document processing
        - Semantic search retrieval
        - Smart chunking strategies
        - Performance metrics
        
        **Technologies:**
        - LangChain
        - LLaMA3 on Groq
        - Sentence Transformers
        - ChromaDB vector store
        """)
        
        st.markdown("Made with ❤️ by Windsurf Engineering")
        
        # System status
        st.markdown(f"System Status: {'🟢 Online' if GROQ_API_KEY else '🔴 API Key Missing'}")

# Display example questions if no conversation yet
if len(st.session_state.messages) <= 1:
    st.subheader("Try asking questions like:")
    st.markdown("- What are the main topics discussed in the document?")
    st.markdown("- Can you summarize the key points from the document?")
    st.markdown("- What information does the document contain about [specific topic]?")
    st.markdown("- How does the document explain [concept or idea]?")

