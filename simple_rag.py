import os
import streamlit as st
from dotenv import load_dotenv
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up directories where files will be stored
DATA_DIR = "data"

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


GROQ_API_KEY = "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"

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file")
    st.error("Error: GROQ_API_KEY not found in .env file. Please add it to your .env file.")
    st.stop()

# Set API key for Groq
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Initialize the LLM using Groq
from langchain_openai import ChatOpenAI

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

# Simple text chunking function for documents
def split_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate the end position for this chunk
        end = min(start + chunk_size, text_length)
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start position for the next chunk, with overlap
        start = start + chunk_size - overlap
        
        if start >= text_length:
            break
            
    return chunks

# Initialize session state for tracking documents
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []

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
    """Process an uploaded document and extract its text content"""
    try:
        # Save the uploaded file to the data directory
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"Saved file to {file_path}")
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Get the appropriate loader for this file type
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            
            # Extract text content
            content = ""
            for doc in documents:
                content += doc.page_content + "\n\n"
            
            # Split into chunks for better handling
            chunks = split_text(content)
            
            # Record the processed document
            doc_info = {
                "filename": uploaded_file.name,
                "type": file_path.split(".")[-1],
                "size": uploaded_file.size,
                "chunks": len(chunks),
                "content": chunks,
                "path": file_path,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in session state to persist across reruns
            st.session_state.processed_documents.append(doc_info)
            
            st.success(f"✅ Successfully processed {uploaded_file.name} into {len(chunks)} text chunks")
            return doc_info
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Document processing error: {e}")
        return None

def process_query(query):
    """Process a user query by finding relevant chunks and using LLM"""
    start_time = time.time()
    
    try:
        if not st.session_state.processed_documents:
            return "Please upload at least one document first.", []
        
        # Collect all chunks from all documents
        all_chunks = []
        all_sources = []
        
        for doc in st.session_state.processed_documents:
            for chunk in doc["content"]:
                all_chunks.append(chunk)
                all_sources.append(doc["filename"])
        
        if not all_chunks:
            return "No document content available to search. Please upload a document.", []
        
        # Very simple relevance by looking for query terms in chunks
        # This is a basic implementation - a real system would use proper embeddings
        relevant_chunks = []
        sources = []
        
        query_terms = query.lower().split()
        for i, chunk in enumerate(all_chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_lower)
            if score > 0:
                relevant_chunks.append((chunk, score, all_sources[i]))
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3 or fewer if less are available
        top_chunks = relevant_chunks[:3]
        
        if not top_chunks:
            # Just take the first chunks if no relevant ones found
            top_chunks = [(all_chunks[i], 0, all_sources[i]) for i in range(min(3, len(all_chunks)))]
        
        # Format for display
        context_chunks = [chunk[0] for chunk in top_chunks]
        for i, (chunk, score, source) in enumerate(top_chunks):
            sources.append({
                "content": chunk[:300] + "...",
                "source": source
            })
        
        # Format documents as context for the LLM
        context = "\n\n".join([f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
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
        
        # Calculate a simple relevance score
        relevance_score = min(1.0, len(top_chunks) / 3.0)
        
        # Record performance metrics
        performance_metrics = {
            "query": query,
            "response_time": response_time,
            "num_chunks": len(top_chunks),
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

# Set up the Streamlit page configuration
st.set_page_config(page_title="✨ AI Knowledge Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling
st.markdown("""
<style>
    /* Main layout styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        padding: 0.5rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stChatMessage:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: rgba(0, 104, 201, 0.1) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Expander styling */
    .streamlit-expander {
        border-radius: 10px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        overflow: hidden !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title and header with modern design
# Only show creator info if no documents have been processed
header_text = "Made by Nandesh Kalashetti" if not st.session_state.processed_documents else ""

st.markdown(f"""
<div style='text-align: center; background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
    <h1 style='color: white; margin: 0; font-size: 2.8rem;'>🧠 AI Knowledge Assistant</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;'>Powered by LLaMA3 on Groq {' | ' + header_text if header_text else ''}</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Simple RAG assistant powered by LLaMA3. Upload documents in the sidebar to get started."}
    ]

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = []

# Create a layout with columns
col1, col2 = st.columns([3, 1])

# Main chat area
with col1:
    # Add a subtle background and border to the chat container
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.02); border-radius: 15px; padding: 0.5rem; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 1rem;">
        <h3 style="margin: 0.5rem 1rem; opacity: 0.8; font-size: 1.2rem; display: flex; align-items: center;">
            <span style="margin-right: 0.5rem; font-size: 1.4rem;">💬</span> Chat with Documents
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with subtle styling
    chat_container = st.container()
    with chat_container:
        # Display chat messages with enhanced styling
        for message in st.session_state.messages:
            icon = "🧠" if message["role"] == "assistant" else "👤"  # Add emoji icons for roles
            with st.chat_message(message["role"]):
                st.markdown(f"{icon} {message['content']}")
    
    # Chat input with enhanced styling 
    st.markdown("""
    <style>
    .stChatInputContainer {
        padding-top: 1rem !important;
        border-top: 1px solid rgba(128, 128, 128, 0.1) !important;
    }
    .stChatInputContainer > div > textarea {
        border-radius: 20px !important;
        padding: 12px 20px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease !important;
    }
    .stChatInputContainer > div > textarea:focus {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        border-color: #3a7bd5 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if query := st.chat_input("✨ Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"👤 {query}")
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Add loading animation
            with st.spinner("Searching knowledge base and generating response..."):
                # Process query and get response
                start_time = time.time()
                answer, sources = process_query(query)
                end_time = time.time()
                
                # Record performance metrics
                response_time = end_time - start_time
                st.session_state.performance_metrics.append({
                    "query": query,
                    "response_time": response_time
                })
                
                # Display answer with AI emoji
                st.markdown(f"🧠 {answer}")
                
                # Show sources if available with simplified styling
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.code(source['content'], language=None)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar for document uploads and info
with st.sidebar:
    # Sidebar Header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #3a7bd5, #00d2ff); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0; text-align: center;'>📄 Document Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple text explanation instead of complex HTML that might render incorrectly
    st.markdown("### 🔍 How It Works")
    st.write("Upload your documents and ask questions about their content!")
    
    # Create a simple grid for file types using columns
    col_ft1, col_ft2, col_ft3, col_ft4 = st.columns(4)
    with col_ft1:
        st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>🔤</h2><p style='margin: 0;'>TXT</p></div>", unsafe_allow_html=True)
    with col_ft2:
        st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>📊</h2><p style='margin: 0;'>CSV</p></div>", unsafe_allow_html=True)
    with col_ft3:
        st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>📝</h2><p style='margin: 0;'>DOCX</p></div>", unsafe_allow_html=True)
    with col_ft4:
        st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>📑</h2><p style='margin: 0;'>PDF</p></div>", unsafe_allow_html=True)
    
    st.caption("Files are saved to ./data/ directory")
    
    # File uploader with enhanced styling
    st.markdown("<h3 style='margin-bottom: 0.5rem;'>📤 Upload Documents</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", 
                                    type=["pdf", "docx", "txt", "csv"],
                                    help="Upload PDF, DOCX, TXT, or CSV files")
    
    if uploaded_file is not None:
        # Custom button with better styling
        st.markdown("""
        <div style='display: flex; justify-content: center;'>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Process Document", use_container_width=True):
            doc_info = process_document(uploaded_file)
    
    # Show processed documents with modern styling
    if st.session_state.processed_documents:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #3a7bd5, #3a6073); padding: 0.8rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: white; margin: 0; text-align: center;'>📜 Knowledge Base</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, doc in enumerate(st.session_state.processed_documents):
            # Document type icon based on file extension
            doc_icon = {
                "pdf": "📑",
                "docx": "📝",
                "txt": "🔤",
                "csv": "📊"
            }.get(doc['type'].lower(), "📄")
            
            with st.expander(f"{doc_icon} {doc['filename']}"):
                # Use simpler Streamlit components instead of complex HTML
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown(f"**Type:** {doc['type'].upper()}")
                    st.markdown(f"**Chunks:** {doc['chunks'] if 'chunks' in doc else 'N/A'}")
                    
                with col_info2:
                    file_size_kb = round(doc['size'] / 1024, 2) if 'size' in doc else 'N/A'
                    st.markdown(f"**Size:** {file_size_kb} KB")
                    
                    # Format date nicely
                    if 'timestamp' in doc:
                        date_added = doc['timestamp'].split('T')[0]
                    else:
                        date_added = 'N/A'
                    st.markdown(f"**Added:** {date_added}")
                
                # File location
                st.markdown(f"**Path:** `{doc['path']}`")
    else:
        # Empty state with a simpler, more reliable display
        st.markdown("### 📂 No Documents Yet")
        st.info("Upload a document to get started")
        st.markdown("")
        # Add some visual space
        st.markdown("<br>", unsafe_allow_html=True)

# Stats and metrics column
with col2:
    # Simple dashboard header
    st.header("📊 Dashboard")
    
    # Clean metrics display with standard Streamlit components
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # Documents metric
        doc_count = len(st.session_state.processed_documents)
        st.metric(label="📚 Documents", value=doc_count)
    
    with col_m2:
        # Queries metric
        num_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric(label="🔎 Queries", value=num_queries)
    
    # Show performance metrics if available with standard Streamlit components
    if "performance_metrics" in st.session_state and st.session_state.performance_metrics:
        st.subheader("⚡ Performance")
        
        col_m3, col_m4 = st.columns(2)
        
        with col_m3:
            # Average response time
            avg_time = sum(m["response_time"] for m in st.session_state.performance_metrics) / len(st.session_state.performance_metrics)
            st.metric(label="⏱ Avg. Response", value=f"{avg_time:.2f}s")
            
        with col_m4:
            # Latest query time
            latest = st.session_state.performance_metrics[-1]
            st.metric(label="🔥 Last Query", value=f"{latest['response_time']:.2f}s")
    
    # About section with simplified styling using Streamlit components
    with st.expander("✨ About"):
        # App title and description
        st.title("🧠 AI Knowledge Assistant")
        st.caption("Intelligent document analysis with RAG")
        
        # App description
        st.markdown("This application uses a simplified Retrieval-Augmented Generation (RAG) approach that avoids complex vector embeddings to deliver reliable document analysis.")
        
        # Features in a cleaner format
        st.subheader("Features")
        features_col1, features_col2 = st.columns(2)
        with features_col1:
            st.markdown("📚 Multi-document support")
            st.markdown("🔍 Text matching retrieval")
        with features_col2:
            st.markdown("📍 Source citation")
            st.markdown("📦 File storage")
        
        # Tech stack
        st.subheader("💻 Tech Stack")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.markdown("🦙 LLaMA3")
            st.markdown("⛓️ LangChain")
        with tech_col2:
            st.markdown("🚀 Groq")
            st.markdown("🔧 Streamlit")
        
        # Only show creator information if no documents are processed
        if not st.session_state.processed_documents:
            st.subheader("👨‍💻 Creator")
            
            col_creator1, col_creator2 = st.columns([1, 3])
            with col_creator1:
                st.markdown("👤")
            with col_creator2:
                st.markdown("**Nandesh Kalashetti**")
                st.caption("Full-Stack Web & Gen-AI Developer")
            
            st.markdown("[Portfolio](https://nandesh-kalashetti.netlify.app) | [Contact](mailto:nandeshkalshetti1@gmail.com)")
            st.markdown("---")
        
        # System status with standard Streamlit components
        st.subheader("System Status")
        
        status_text = "🟢 Online" if GROQ_API_KEY else "🔴 API Key Missing"
        st.info(status_text)

# Tips for new users with standard Streamlit components
if len(st.session_state.messages) <= 1:
    st.header("🚀 Getting Started")
    
    step1, step2, step3 = st.columns(3)
    
    with step1:
        st.markdown("**Step 1**")
        st.markdown("Upload your documents using the sidebar uploader")
        
    with step2:
        st.markdown("**Step 2**")
        st.markdown("Click '🔄 Process Document' to extract text content")
        
    with step3:
        st.markdown("**Step 3**")
        st.markdown("Ask questions about your documents in the chat box")
    
    st.info("💾 Your files are saved to the `./data/` directory in this project")
    st.markdown("---")
