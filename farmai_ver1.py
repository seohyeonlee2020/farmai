import streamlit as st
import os
import json
import time
import requests
import logging
import warnings

# CRITICAL: Set environment variables BEFORE any other imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOBUF_PYTHON_IMPLEMENTATION"] = "python"

# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=FutureWarning)

# Import text preprocessing utility
try:
    from utils.text_data_preprocessing import extract_text
except ImportError as e:
    st.error(f"Error importing text preprocessing utilities: {e}")
    st.stop()

# Import LangChain components with error handling
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error(f"Error importing LangChain components: {e}")
    st.info("Please ensure all required packages are installed. See requirements below.")
    st.code("""
    pip install langchain
    pip install langchain-community
    pip install langchain-huggingface
    pip install langchain-text-splitters
    pip install faiss-cpu
    pip install sentence-transformers
    pip install protobuf==3.20.3
    """)
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)


@st.cache_data
def load_text_data():
    """Load text data from JSON file or extract from directory"""
    try:
        if not os.path.exists("./text_data.json"):
            directory = "./farmai_training_data"
            if not os.path.exists(directory):
                st.error(f"Training data directory '{directory}' not found!")
                return {}

            text_data = extract_text(directory)
            with open('text_data.json', 'w', encoding='utf-8') as fp:
                json.dump(text_data, fp, ensure_ascii=False, indent=2)
            return text_data
        else:
            with open('text_data.json', 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                return text_data
    except Exception as e:
        st.error(f"Error loading text data: {e}")
        logging.error(f"Error in load_text_data: {str(e)}")
        return {}


def dict_to_documents(file_dict):
    """
    Convert a dictionary of {filename: content} to LangChain Document objects

    Args:
        file_dict (dict): Dictionary with filename as key, text content as value

    Returns:
        list: List of Document objects
    """
    documents = []

    if not file_dict:
        logging.warning("Empty file dictionary provided to dict_to_documents")
        return documents

    for filename, content in file_dict.items():
        if not content or not content.strip():
            logging.warning(f"Skipping empty content for file: {filename}")
            continue

        doc = Document(
            page_content=str(content).strip(),
            metadata={
                "source": filename,
                "filename": os.path.basename(filename),
                "file_extension": os.path.splitext(filename)[1],
                "char_count": len(content),
                "word_count": len(content.split())
            }
        )
        documents.append(doc)

    logging.info(f"Created {len(documents)} documents from {len(file_dict)} files")
    return documents


@st.cache_data
def prepare_documents():
    """Convert text data to chunked documents"""
    try:
        text_data = load_text_data()
        if not text_data:
            st.error("No text data available for processing")
            return []

        data = dict_to_documents(text_data)
        if not data:
            st.error("No valid documents created from text data")
            return []

        # Use smaller chunks for better retrieval with overlapping
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Increased chunk size for better context
            chunk_overlap=50,  # Increased overlap for continuity
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
            length_function=len
        )

        split_docs = text_splitter.split_documents(data)
        logging.info(f"Split {len(data)} documents into {len(split_docs)} chunks")
        return split_docs

    except Exception as e:
        st.error(f"Error preparing documents: {str(e)}")
        logging.error(f"Error in prepare_documents: {str(e)}")
        return []


@st.cache_resource
def create_vectorstore():
    """Create and return vectorstore with embeddings"""
    try:
        # Get prepared documents
        documents = prepare_documents()
        if not documents:
            raise ValueError("No documents available to create vectorstore")

        # Initialize embeddings with explicit device configuration
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

        logging.info(f"Creating FAISS vectorstore with {len(documents)} documents")

        # Create vectorstore with error handling
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )

        logging.info("Successfully created FAISS vectorstore")
        return vectorstore

    except Exception as e:
        error_msg = f"Error creating vectorstore: {str(e)}"
        st.error(error_msg)
        logging.error(error_msg)
        raise


def load_vectorstore():
    """Initialize vectorstore once per session with proper error handling"""
    try:
        if "vectorstore" not in st.session_state:
            with st.spinner("🔄 Loading knowledge base... This may take a moment."):
                st.session_state.vectorstore = create_vectorstore()
            st.success("✅ Knowledge base loaded successfully!")
        return True
    except Exception as e:
        error_msg = f"Failed to load vectorstore: {str(e)}"
        st.error(error_msg)
        logging.error(error_msg)

        # Provide debugging info
        with st.expander("🔍 Debug Information"):
            st.write("**Possible solutions:**")
            st.write("1. Check if training data directory exists")
            st.write("2. Verify protobuf version compatibility")
            st.write("3. Restart the application")
            st.write("4. Clear Streamlit cache")
        return False


@st.cache_data
def load_prompt_template():
    """Load prompt template from file with fallback"""
    try:
        template_path = 'utils/prompt_template.txt'
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            logging.warning(f"Prompt template file not found: {template_path}")
            # Return default template
            return """Based on the following farming knowledge context, provide a helpful and accurate answer to the user's question.

Context:
{combined_context}

Question: {user_question}

Answer: Provide a clear, practical answer based on the context above. If the context doesn't contain relevant information, acknowledge this and provide general farming guidance where appropriate."""

    except Exception as e:
        logging.error(f"Error loading prompt template: {str(e)}")
        return "Context: {combined_context}\n\nQuestion: {user_question}\n\nAnswer:"


def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
        else:
            return False, f"Ollama responded with status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama (connection refused)"
    except requests.exceptions.Timeout:
        return False, "Ollama request timed out"
    except Exception as e:
        return False, f"Ollama check failed: {str(e)}"


def call_ollama(prompt, model):
    """Send prompt to Ollama API with improved error handling"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 512
                }
            },
            timeout=60  # Increased timeout for model processing
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response generated')
        else:
            return f"Ollama API Error: HTTP {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. The model might be processing a complex query."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please ensure Ollama is running."
    except Exception as e:
        return f"Chatbot Error: {str(e)}"


# Streamlit App Configuration
st.set_page_config(
    page_title="FarmAI: Climate-Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main App
st.title("🌱 FarmAI: Climate-Smart Farming Assistant")
st.markdown("*Your AI-powered companion for sustainable farming practices*")

# Check Ollama status in sidebar
with st.sidebar:
    st.header("System Status")
    ollama_running, ollama_status = check_ollama_status()

    if ollama_running:
        st.success(f"✅ {ollama_status}")
    else:
        st.error(f"❌ {ollama_status}")
        st.info("Start Ollama with: `ollama serve`")

    # Add cache management
    st.header("Cache Management")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.success("Cache cleared!")
        st.rerun()

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Ask your farming question:",
        placeholder="e.g., How can I improve soil health for tomato cultivation?",
        help="Ask about climate-smart farming practices, crop management, soil health, etc."
    )

with col2:
    model_choice = st.selectbox(
        "Model:",
        ["qwen2:0.5b", "qwen3:0.6b", "tinyllama"],
        index=0
    )

# Process user input
if user_question:
    if not ollama_running:
        st.warning("⚠️ Ollama is not running. Please start Ollama to get responses.")
    else:
        try:
            # Load vectorstore with proper error handling
            if not load_vectorstore():
                st.error("❌ Failed to initialize the knowledge base. Please check the logs and try again.")
                st.stop()

            logging.info(f"Processing user question: {user_question}")

            # Create progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Retrieve relevant documents
            status_text.text("🔍 Searching for relevant documents...")
            progress_bar.progress(25)

            retrieved_docs = st.session_state.vectorstore.similarity_search(
                user_question,
                k=3  # Get more documents for better context
            )

            logging.info(f"Retrieved {len(retrieved_docs)} documents")

            # Step 2: Process documents
            progress_bar.progress(50)
            status_text.text("📄 Processing retrieved documents...")

            if not retrieved_docs:
                st.warning("❓ No relevant documents found. Try rephrasing your question or using more specific terms.")
                logging.warning("No documents retrieved for user question")
            else:
                # Create context from retrieved docs
                context_texts = "\n\n".join([
                    f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                    for doc in retrieved_docs
                ])

                # Step 3: Generate response
                progress_bar.progress(75)
                status_text.text("🤖 Generating AI response...")

                # Load and format prompt
                prompt_template = load_prompt_template()
                prompt = prompt_template.format(
                    combined_context=context_texts,
                    user_question=user_question
                )

                # Generate response
                logging.info(f"Generating response with model: {model_choice}")

                start_time = time.time()
                llm_response = call_ollama(prompt, model_choice)
                end_time = time.time()

                # Complete progress
                progress_bar.progress(100)
                status_text.text("✅ Response generated!")

                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # Display results
                response_time = end_time - start_time
                logging.info(f"Total response time: {response_time:.2f} seconds")

                # Show response
                st.markdown("### 🎯 Answer")
                st.markdown(llm_response)

                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Response Time", f"{response_time:.2f}s")
                with col2:
                    st.metric("📚 Sources Found", len(retrieved_docs))
                with col3:
                    st.metric("🤖 Model Used", model_choice)

                # Show debug info in expandable section
                with st.expander("🔍 Source Documents & Debug Info"):
                    st.markdown("**Retrieved Documents:**")
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.container():
                            st.markdown(f"**📄 Document {i}: {doc.metadata.get('filename', 'Unknown')}**")
                            st.markdown(f"*Characters: {doc.metadata.get('char_count', 'N/A')} | Words: {doc.metadata.get('word_count', 'N/A')}*")
                            st.markdown(f"```\n{doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}\n```")
                            st.markdown("---")

        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            logging.error(error_msg)
            st.error(f"❌ {error_msg}")

            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common solutions:**
                1. **Restart the application** - Sometimes a fresh start helps
                2. **Clear the cache** - Use the sidebar button to clear cached data
                3. **Check Ollama** - Ensure Ollama is running and responsive
                4. **Verify data files** - Ensure training data directory exists
                5. **Update dependencies** - Make sure all packages are up to date
                """)

# Footer
st.markdown("---")
st.markdown(
    "🌱 **FarmAI** - Powered by AI for sustainable agriculture | "
    f"Built with Streamlit • LangChain • FAISS • Ollama"
)
