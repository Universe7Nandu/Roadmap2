# Advanced RAG Assistant

A fully-featured Retrieval-Augmented Generation (RAG) system that provides accurate answers by combining document retrieval with LLM text generation.

## Features

- **Multi-format Document Processing**: Support for PDF, DOCX, TXT, XLSX, CSV, and web content using unstructured.io
- **Advanced Embedding Models**: Integration with HuggingFace, Groq, and Mistral embedding models
- **Vector Database Storage**: Uses Chroma, FAISS, and Qdrant for efficient vector storage and retrieval
- **Sophisticated Chunking**: Smart document segmentation with customizable chunk sizes and overlap
- **Enhanced Retrieval Techniques**: Multi-query generation, contextual compression, and hybrid search
- **Memory Management**: Session-based and persistent memory for maintaining conversation context
- **Evaluation Metrics**: Integrated evaluation using TruLens-inspired metrics for relevance, faithfulness, and quality
- **Hallucination Detection**: Automatic detection and flagging of potential hallucinations
- **Modern UI**: Clean, responsive interface built with Streamlit

## Getting Started

### Prerequisites

- Python 3.9+
- Groq API Key

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key
```

### Running the Application

```bash
streamlit run app.py
```

### Usage

1. Upload documents (PDF, DOCX, TXT, XLSX) or provide web URLs
2. Ask questions about your documents
3. Review the generated answers with source citations
4. Examine evaluation metrics and hallucination analysis

## System Architecture

The system follows a modular architecture with the following components:

- **Document Processor**: Handles document loading and chunking
- **Embedding Manager**: Manages different embedding models
- **Vector Store Manager**: Interfaces with vector databases
- **Retrieval Manager**: Implements various retrieval strategies
- **Memory Manager**: Handles conversation history and context
- **Evaluation Module**: Provides metrics and performance tracking

## Evaluation Metrics

The system tracks the following key metrics:

- **Context Relevance**: How relevant retrieved documents are to the query
- **Answer Faithfulness**: How accurately the response reflects source documents
- **Answer Relevance**: How well the response addresses the user's query
- **RAG Triad Score**: Combined metric of overall RAG performance

## License

This project is open source and available under the MIT License.

## Acknowledgements

- LangChain for the foundational RAG components
- Groq for providing high-performance LLaMA3 API access
- Streamlit for the interactive UI framework
- HuggingFace for embedding models

# RAG Streamlit Application

This application demonstrates a Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and Groq.

## Features

- Upload documents (PDF, DOCX, TXT, CSV)
- Process and index document content
- Add web pages by URL
- Ask questions about your documents
- View document statistics and management
- Beautiful Streamlit UI

## Setup

1. Clone this repository:
```bash
git clone https://github.com/Universe7Nandu/Roadmap2.git
cd Roadmap2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your Groq API key:
```
GROQ_API_KEY="your-groq-api-key"
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment

This application is ready for deployment on Streamlit Cloud:

1. Push to GitHub
2. Connect your repository on [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with the command: `streamlit run app.py`

## Structure

- `app.py`: Main application file
- `requirements.txt`: Python dependencies
- `packages.txt`: System dependencies for document processing
- `.streamlit/config.toml`: Streamlit configuration
- `data/`: Directory for uploaded documents
- `vectordb/`: Vector database storage

## License

MIT
