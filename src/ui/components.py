import streamlit as st
from typing import List, Dict, Any, Optional, Callable
import time
import base64
from pathlib import Path

class StreamlitUI:
    """
    A class for creating a modern UI for the RAG assistant using Streamlit.
    Provides various UI components and styling.
    """
    
    @staticmethod
    def set_page_config(title: str = "RAG Assistant", layout: str = "wide"):
        """Configure the Streamlit page with title and layout."""
        st.set_page_config(
            page_title=title,
            page_icon="📚",
            layout=layout,
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS
        st.markdown("""
            <style>
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
                .stApp {
                    background-color: #f5f7f9;
                }
                .chat-message {
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    display: flex;
                    flex-direction: row;
                    align-items: flex-start;
                }
                .chat-message.user {
                    background-color: #e6f3ff;
                }
                .chat-message.assistant {
                    background-color: #f0f2f6;
                }
                .chat-message .avatar {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    object-fit: cover;
                    margin-right: 1rem;
                }
                .chat-message .message {
                    flex-grow: 1;
                }
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                .stTextInput>div>div>input {
                    border-radius: 4px;
                }
                .source-documents {
                    font-size: 0.85rem;
                    background-color: #f8f9fa;
                    border-radius: 0.5rem;
                    padding: 0.5rem;
                    margin-top: 0.5rem;
                }
                .source-documents-title {
                    font-weight: bold;
                    font-size: 0.9rem;
                    margin-bottom: 0.3rem;
                }
                .model-info {
                    font-size: 0.7rem;
                    color: #708090;
                    text-align: right;
                }
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    height: calc(100vh - 200px);
                    overflow-y: auto;
                    padding: 1rem;
                    background-color: white;
                    border-radius: 0.5rem;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                }
                .info-box {
                    background-color: #e7f5ff;
                    border-left: 4px solid #1e88e5;
                    padding: 1rem;
                    border-radius: 0.25rem;
                }
                .warning-box {
                    background-color: #fff8e1;
                    border-left: 4px solid #ffc107;
                    padding: 1rem;
                    border-radius: 0.25rem;
                }
                .performance-metrics {
                    background-color: #f1f8e9;
                    border-radius: 0.5rem;
                    padding: 0.5rem;
                    margin-top: 0.5rem;
                }
                .metrics-table {
                    font-size: 0.8rem;
                    width: 100%;
                }
                .metrics-table th {
                    text-align: left;
                    padding: 0.3rem;
                }
                .metrics-table td {
                    padding: 0.3rem;
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header(title: str = "📚 GenAI RAG Assistant", subtitle: str = "Powered by LLaMA3 on Groq"):
        """Render the page header with title and subtitle."""
        st.markdown(f"<h1 style='text-align: center; margin-bottom: 0;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: #708090; margin-top: 0;'>{subtitle}</h4>", unsafe_allow_html=True)
    
    @staticmethod
    def init_session_state():
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents" not in st.session_state:
            st.session_state.documents = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "evaluation_results" not in st.session_state:
            st.session_state.evaluation_results = {}
    
    @staticmethod
    def file_uploader(
        accepted_types: List[str] = ["pdf", "docx", "txt", "xlsx"],
        key: str = "file_uploader",
        on_upload: Optional[Callable] = None
    ):
        """Create a file uploader component."""
        uploaded_file = st.sidebar.file_uploader(
            "Upload a Document",
            type=accepted_types,
            key=key
        )
        
        if uploaded_file and on_upload:
            on_upload(uploaded_file)
            
        return uploaded_file
    
    @staticmethod
    def document_explorer(documents: List[Dict[str, Any]]):
        """Create a document explorer component."""
        if not documents:
            st.sidebar.info("No documents uploaded yet.")
            return
            
        st.sidebar.markdown("### Document Explorer")
        
        for i, doc in enumerate(documents):
            with st.sidebar.expander(f"{doc.get('name', f'Document {i+1}')} ({doc.get('type', 'Unknown')})"):
                st.write(f"Size: {doc.get('size', 'Unknown')}")
                if doc.get('metadata'):
                    st.write("Metadata:")
                    st.json(doc['metadata'])
                    
                st.button(
                    "Remove",
                    key=f"remove_doc_{i}",
                    on_click=lambda idx=i: StreamlitUI._remove_document(idx)
                )
    
    @staticmethod
    def _remove_document(index: int):
        """Remove a document from the session state."""
        if 0 <= index < len(st.session_state.documents):
            st.session_state.documents.pop(index)
    
    @staticmethod
    def url_input(on_submit: Optional[Callable] = None):
        """Create a URL input component."""
        with st.sidebar.form("url_form", clear_on_submit=True):
            url = st.text_input("Enter a URL to process:", placeholder="https://example.com")
            submitted = st.form_submit_button("Process URL")
            
        if submitted and url and on_submit:
            on_submit(url)
            
        return url if submitted and url else None
    
    @staticmethod
    def chat_container():
        """Create a container for chat messages."""
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                StreamlitUI.chat_message(
                    role=message["role"],
                    content=message["content"],
                    avatar=message.get("avatar"),
                    sources=message.get("sources")
                )
                
        return chat_container
    
    @staticmethod
    def chat_message(
        role: str,
        content: str,
        avatar: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Render a single chat message with optional sources."""
        # Default avatars
        user_avatar = "👤"
        assistant_avatar = "🤖"
        
        with st.container():
            # Create message container with appropriate styling
            st.markdown(
                f"""
                <div class="chat-message {role}">
                    <div class="avatar">
                        {avatar if avatar else user_avatar if role == "user" else assistant_avatar}
                    </div>
                    <div class="message">
                        {content}
                        
                        {StreamlitUI._render_sources(sources) if sources else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    @staticmethod
    def _render_sources(sources: List[Dict[str, Any]]) -> str:
        """Render source documents HTML."""
        if not sources:
            return ""
            
        sources_html = """
            <div class="source-documents">
                <div class="source-documents-title">Sources:</div>
                <ul style="margin-top: 0; padding-left: 1.5rem;">
        """
        
        for source in sources:
            sources_html += f"""
                <li>
                    <strong>{source.get('title', 'Document')}</strong>
                    {f" (Page {source.get('page', '')})" if source.get('page') else ""}
                </li>
            """
            
        sources_html += """
                </ul>
            </div>
        """
        
        return sources_html
    
    @staticmethod
    def chat_input(on_submit: Callable):
        """Create a chat input component."""
        with st.container():
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
            # Create a form to handle the chat input
            with st.form(key="chat_form", clear_on_submit=True):
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    user_input = st.text_input(
                        "Ask a question:",
                        placeholder="What would you like to know about the documents?",
                        label_visibility="collapsed"
                    )
                    
                with col2:
                    submit_button = st.form_submit_button("Send")
                    
            if submit_button and user_input:
                on_submit(user_input)
    
    @staticmethod
    def add_message(role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
        """Add a message to the chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        if sources:
            message["sources"] = sources
            
        st.session_state.messages.append(message)
        
        # Also add to chat history for the RAG system
        if role == "user":
            st.session_state.chat_history.append({"type": "human", "content": content})
        else:
            st.session_state.chat_history.append({"type": "ai", "content": content})
    
    @staticmethod
    def display_thinking(message: str = "Thinking..."):
        """Display a thinking indicator while processing."""
        with st.spinner(message):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(f"<div class='info-box'>{message}</div>", unsafe_allow_html=True)
            return thinking_placeholder
    
    @staticmethod
    def display_metrics(metrics: Dict[str, Any]):
        """Display RAG system performance metrics."""
        st.sidebar.markdown("### Performance Metrics")
        
        with st.sidebar.expander("View Metrics", expanded=False):
            if not metrics:
                st.info("No metrics available yet.")
                return
                
            st.markdown(
                """
                <div class="performance-metrics">
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Relevance</td>
                            <td>{:.2f}</td>
                        </tr>
                        <tr>
                            <td>Faithfulness</td>
                            <td>{:.2f}</td>
                        </tr>
                        <tr>
                            <td>Response Time</td>
                            <td>{:.2f}s</td>
                        </tr>
                    </table>
                </div>
                """.format(
                    metrics.get("relevance", 0),
                    metrics.get("faithfulness", 0),
                    metrics.get("response_time", 0)
                ),
                unsafe_allow_html=True
            )
    
    @staticmethod
    def display_document_preview(document: Dict[str, Any]):
        """Display a preview of the uploaded document."""
        if not document:
            return
            
        st.sidebar.markdown("### Document Preview")
        
        with st.sidebar.expander("View Preview", expanded=False):
            if document.get("type") == "pdf":
                # Display PDF preview
                if document.get("content"):
                    preview = document["content"][:500] + "..." if len(document["content"]) > 500 else document["content"]
                    st.markdown(f"```\n{preview}\n```")
            elif document.get("type") in ["docx", "txt"]:
                # Display text preview
                if document.get("content"):
                    preview = document["content"][:500] + "..." if len(document["content"]) > 500 else document["content"]
                    st.markdown(f"```\n{preview}\n```")
            else:
                st.info("Preview not available for this document type.")
    
    @staticmethod
    def display_evaluation_results(eval_results: Dict[str, Any]):
        """Display RAG evaluation results."""
        if not eval_results:
            return
            
        st.sidebar.markdown("### Evaluation Results")
        
        with st.sidebar.expander("View Evaluation", expanded=False):
            # Display TruLens-style metrics
            if "rag_triad" in eval_results:
                st.markdown("#### RAG Triad Score")
                st.progress(eval_results["rag_triad"])
                st.markdown(f"{eval_results['rag_triad']:.2f}/1.0")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Context Relevance**")
                    st.progress(eval_results.get("context_relevance", 0))
                    st.markdown(f"{eval_results.get('context_relevance', 0):.2f}/1.0")
                    
                with col2:
                    st.markdown("**Answer Faithfulness**")
                    st.progress(eval_results.get("answer_faithfulness", 0))
                    st.markdown(f"{eval_results.get('answer_faithfulness', 0):.2f}/1.0")
                
                st.markdown("**Answer Relevance**")
                st.progress(eval_results.get("answer_relevance", 0))
                st.markdown(f"{eval_results.get('answer_relevance', 0):.2f}/1.0")
            
            # Display hallucination analysis if available
            if eval_results.get("hallucination_analysis"):
                analysis = eval_results["hallucination_analysis"]
                if analysis.get("has_hallucinations"):
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <strong>Potential Hallucinations Detected</strong> (Confidence: {analysis.get('confidence', 0):.2f})
                            <ul>
                                {"".join([f"<li>{statement}</li>" for statement in analysis.get("hallucinated_statements", [])])}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="info-box">
                            <strong>No Hallucinations Detected</strong> (Confidence: {analysis.get('confidence', 0):.2f})
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    @staticmethod
    def loading_animation():
        """Display a loading animation while processing documents."""
        with st.spinner("Processing document..."):
            # Create a simple animation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            st.success("Document processed successfully!")
            time.sleep(0.5)
            progress_bar.empty()
    
    @staticmethod
    def display_file_info(file_info: Dict[str, Any]):
        """Display information about the uploaded file."""
        if not file_info:
            return
            
        st.sidebar.markdown("### File Information")
        st.sidebar.markdown(f"**Name:** {file_info.get('name', 'Unknown')}")
        st.sidebar.markdown(f"**Type:** {file_info.get('type', 'Unknown')}")
        st.sidebar.markdown(f"**Size:** {file_info.get('size', 'Unknown')} bytes")
        
        if file_info.get("chunks"):
            st.sidebar.markdown(f"**Chunks:** {len(file_info['chunks'])}")
    
    @staticmethod
    def display_error(error_message: str):
        """Display an error message."""
        st.error(error_message)
        
    @staticmethod
    def display_success(success_message: str):
        """Display a success message."""
        st.success(success_message)
        
    @staticmethod
    def display_info(info_message: str):
        """Display an info message."""
        st.info(info_message)
        
    @staticmethod
    def display_warning(warning_message: str):
        """Display a warning message."""
        st.warning(warning_message)
        
    @staticmethod
    def get_css_file_content(filepath: str) -> str:
        """Load CSS content from a file."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading CSS file: {e}")
            return ""
            
    @staticmethod
    def get_file_download_link(file_path: str, link_text: str = "Download File"):
        """Generate a download link for a file."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            file_name = Path(file_path).name
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
            return href
        except Exception as e:
            print(f"Error creating download link: {e}")
            return ""
