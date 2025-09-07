import streamlit as st
import os
import json
import time
import logging
import warnings
import hashlib

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY= os.getenv('GROQ_API_KEY')

# Set environment variables BEFORE imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOBUF_PYTHON_IMPLEMENTATION"] = "python"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=FutureWarning)

# Import text preprocessing utility
try:
    from utils.text_data_preprocessing import extract_text
except ImportError as e:
    st.error(f"Error importing text preprocessing utilities: {e}")
    st.info("Please ensure utils/text_data_preprocessing.py exists")
    st.stop()

# Import required packages
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from groq import Groq
except ImportError as e:
    st.error(f"Error importing required packages: {e}")
    st.code("""
    # Install required packages:
    pip install langchain langchain-community langchain-huggingface
    pip install langchain-text-splitters faiss-cpu sentence-transformers
    pip install protobuf==3.20.3 groq
    """)
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# =============================================================================
# GROQ INTEGRATION FUNCTIONS
# =============================================================================

def get_groq_client():
    """Initialize Groq client with API key"""
    try:
        # Try to get API key from Streamlit secrets first, then environment
        api_key = None
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        else:
            api_key = GROQ_API_KEY

        if not api_key:
            return None, "No API key found"

        client = Groq(api_key=api_key)
        return client, "Connected"

    except Exception as e:
        return None, f"Error: {str(e)}"

def get_available_models():
    """Get list of available Groq models"""
    return [
        "llama-3.1-70b-versatile",  # Great for complex reasoning
        "llama-3.1-8b-instant",     # Fast and efficient
        "mixtral-8x7b-32768",       # Good for long contexts
        "gemma-7b-it",              # Google's model
        "llama3-70b-8192",          # Alternative Llama model
        "llama3-8b-8192"            # Smaller, faster option
    ]

def call_groq_llm(prompt, model="llama-3.1-8b-instant", max_tokens=1024):
    """
    Call Groq API for text generation

    Args:
        prompt (str): Input prompt
        model (str): Model name
        max_tokens (int): Maximum tokens to generate

    Returns:
        str: Generated response
    """
    try:
        client, status = get_groq_client()
        if not client:
            return f"❌ Groq API Error: {status}"

        # Create chat completion
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agricultural advisor specializing in climate-smart farming practices. Provide practical, evidence-based advice to help farmers improve their agricultural operations sustainably."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=None
        )

        response = completion.choices[0].message.content
        return response if response else "No response generated"

    except Exception as e:
        error_msg = f"Groq API Error: {str(e)}"
        logging.error(error_msg)

        # Provide helpful error messages
        if "401" in str(e):
            return "❌ Invalid API key. Please check your Groq API key."
        elif "429" in str(e):
            return "❌ Rate limit exceeded. Please wait a moment and try again."
        elif "quota" in str(e).lower():
            return "❌ API quota exceeded. Please check your Groq account limits."
        else:
            return f"❌ {error_msg}"

# =============================================================================
# VECTORSTORE AND DOCUMENT PROCESSING
# =============================================================================

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
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading text data: {e}")
        logging.error(f"Error in load_text_data: {str(e)}")
        return {}

def dict_to_documents(file_dict):
    """Convert dictionary to LangChain Documents"""
    documents = []

    if not file_dict:
        logging.warning("Empty file dictionary provided")
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Larger chunks for better context
            chunk_overlap=100,  # Good overlap for continuity
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

def get_embeddings():
    """Get embeddings instance for consistency"""
    return HuggingFaceEmbeddings(
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

def create_vectorstore_index():
    """Create vectorstore index name based on data content"""
    text_data = load_text_data()
    if not text_data:
        return None

    # Create a simple hash of the data for versioning
    content_hash = hashlib.md5(
        json.dumps(sorted(text_data.items()), ensure_ascii=False).encode()
    ).hexdigest()[:8]

    return f"faiss_index_{content_hash}"

def create_and_save_vectorstore(save_path: str):
    """Create and save vectorstore to disk"""
    try:
        documents = prepare_documents()
        if not documents:
            raise ValueError("No documents available to create vectorstore")

        embeddings = get_embeddings()

        logging.info(f"Creating FAISS vectorstore with {len(documents)} documents")

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Save to disk
        vectorstore.save_local(save_path)
        logging.info(f"Successfully saved FAISS vectorstore to {save_path}")

        return vectorstore

    except Exception as e:
        error_msg = f"Error creating and saving vectorstore: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

def load_persistent_vectorstore():
    """Load or create persistent vectorstore"""
    try:
        vectorstore_dir = "./vectorstore"
        os.makedirs(vectorstore_dir, exist_ok=True)

        index_name = create_vectorstore_index()
        if not index_name:
            raise ValueError("No data available to create index")

        vectorstore_path = os.path.join(vectorstore_dir, index_name)

        # Check if vectorstore exists
        if os.path.exists(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
            try:
                logging.info(f"Loading existing vectorstore from {vectorstore_path}")

                embeddings = get_embeddings()
                vectorstore = FAISS.load_local(
                    vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                logging.info("Successfully loaded existing FAISS vectorstore")
                return vectorstore, True  # True = loaded from disk

            except Exception as e:
                logging.warning(f"Failed to load existing vectorstore: {str(e)}")

        # Create new vectorstore
        logging.info("Creating new vectorstore...")
        vectorstore = create_and_save_vectorstore(vectorstore_path)
        return vectorstore, False  # False = newly created

    except Exception as e:
        error_msg = f"Error with persistent vectorstore: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

@st.cache_resource
def get_vectorstore():
    """Get vectorstore - cached for session"""
    vectorstore, was_loaded = load_persistent_vectorstore()
    return vectorstore, was_loaded

def load_vectorstore():
    """Initialize vectorstore once per session with persistence"""
    try:
        if "vectorstore" not in st.session_state:
            with st.spinner("🔄 Loading knowledge base..."):
                vectorstore, was_loaded = get_vectorstore()
                st.session_state.vectorstore = vectorstore

            if was_loaded:
                st.success("✅ Knowledge base loaded from disk (fast loading)!")
            else:
                st.success("✅ Knowledge base created and saved for future use!")
        return True
    except Exception as e:
        error_msg = f"Failed to load vectorstore: {str(e)}"
        st.error(error_msg)
        logging.error(error_msg)
        return False

def rebuild_vectorstore():
    """Force rebuild of vectorstore"""
    try:
        # Clear session state
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore

        # Clear cache
        st.cache_resource.clear()

        # Remove existing vectorstore files
        vectorstore_dir = "./vectorstore"
        if os.path.exists(vectorstore_dir):
            import shutil
            shutil.rmtree(vectorstore_dir)
            logging.info("Removed existing vectorstore directory")

        # Force recreation
        with st.spinner("🔄 Rebuilding knowledge base from scratch..."):
            vectorstore, _ = get_vectorstore()
            st.session_state.vectorstore = vectorstore

        st.success("✅ Knowledge base rebuilt successfully!")
        return True

    except Exception as e:
        st.error(f"Failed to rebuild vectorstore: {str(e)}")
        return False

@st.cache_data
def load_prompt_template():
    """Load prompt template with fallback"""
    try:
        template_path = 'utils/prompt_template.txt'
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            return """Based on the following farming knowledge context, provide helpful and practical agricultural advice.

Context:
{combined_context}

Farmer's Question: {user_question}

Instructions: Provide a clear, actionable answer based on the context above. Focus on practical steps, sustainable practices, and climate-smart farming techniques. If the context doesn't contain directly relevant information, provide general best practices while noting the limitation.

Answer:"""
    except Exception as e:
        logging.error(f"Error loading prompt template: {str(e)}")
        return "Context: {combined_context}\n\nQuestion: {user_question}\n\nAnswer:"

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(
    page_title="FarmAI: Climate-Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 FarmAI: Climate-Smart Farming Assistant")
st.markdown("*Powered by Groq's lightning-fast AI inference - completely free!*")

# Sidebar for configuration and status
with st.sidebar:
    st.header("🚀 Groq Configuration")

    # Check API key status
    client, status = get_groq_client()
    if client:
        st.success(f"✅ Groq API: {status}")
    else:
        st.error(f"❌ Groq API: {status}")
        st.info("🔑 **Get your free Groq API key:**")
        st.markdown("1. Visit [console.groq.com](https://console.groq.com)")
        st.markdown("2. Sign up/login")
        st.markdown("3. Create API key")
        st.markdown("4. Add to Streamlit secrets or environment")

        with st.expander("How to add API key"):
            st.code("""
# For local development, create .streamlit/secrets.toml:
GROQ_API_KEY = "your_api_key_here"

# For deployment, add to environment variables:
export GROQ_API_KEY="your_api_key_here"
            """)

    # Model selection
    st.header("🤖 Model Settings")
    selected_model = st.selectbox(
        "Choose Groq Model:",
        get_available_models(),
        index=1,  # Default to llama-3.1-8b-instant (fast)
        help="llama-3.1-8b-instant is recommended for speed"
    )

    max_tokens = st.slider(
        "Max Response Length:",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Longer responses use more tokens"
    )

    # Vectorstore management
    st.header("💾 Knowledge Base")

    vectorstore_dir = "./vectorstore"
    if os.path.exists(vectorstore_dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(vectorstore_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        size_mb = total_size / (1024 * 1024)
        st.info(f"💾 Size: {size_mb:.1f} MB")
    else:
        st.info("💾 No vectorstore found")

    # Management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            st.success("Cache cleared!")
            st.rerun()

    with col2:
        if st.button("🔄 Rebuild"):
            if rebuild_vectorstore():
                st.rerun()

    # Data statistics
    st.header("📊 Data Info")
    try:
        text_data = load_text_data()
        if text_data:
            st.info(f"📚 {len(text_data)} source files")
            total_chars = sum(len(content) for content in text_data.values())
            st.info(f"📝 {total_chars:,} characters")
        else:
            st.warning("⚠️ No source data")
    except Exception:
        st.error("❌ Error loading data info")

    # Status indicator
    if "vectorstore" in st.session_state:
        st.success("✅ Knowledge base ready")
    else:
        st.info("⏳ Knowledge base not loaded")

# Main interface
st.markdown("### Ask your farming question:")

user_question = st.text_input(
    "",
    placeholder="e.g., How can I improve soil health for tomato cultivation in changing climate conditions?",
    help="Ask about sustainable farming practices, crop management, soil health, pest control, climate adaptation, etc.",
    label_visibility="collapsed"
)

# Quick example questions
st.markdown("**💡 Example questions:**")
example_questions = [
    "How to improve soil health for vegetable crops?",
    "What are climate-resilient farming practices?",
    "How to manage pests naturally without chemicals?",
    "Best practices for water conservation in agriculture?"
]

cols = st.columns(len(example_questions))
for i, question in enumerate(example_questions):
    if cols[i].button(f"📝 {question[:25]}...", key=f"ex_{i}"):
        st.session_state.example_question = question
        st.rerun()

# Handle example question selection
if hasattr(st.session_state, 'example_question'):
    user_question = st.session_state.example_question
    delattr(st.session_state, 'example_question')

# Process user input
if user_question:
    if not client:
        st.error("❌ Please configure your Groq API key to use the assistant.")
        st.stop()

    try:
        # Load vectorstore
        if not load_vectorstore():
            st.error("❌ Failed to initialize knowledge base")
            st.stop()

        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Step 1: Search knowledge base
        status_text.text("🔍 Searching knowledge base...")
        progress_bar.progress(25)

        retrieved_docs = st.session_state.vectorstore.similarity_search(
            user_question,
            k=4  # Get more documents for comprehensive context
        )

        logging.info(f"Retrieved {len(retrieved_docs)} documents")

        # Step 2: Prepare context
        progress_bar.progress(50)
        status_text.text("📄 Preparing context...")

        if not retrieved_docs:
            st.warning("❓ No relevant documents found in knowledge base. Providing general guidance...")
            context_texts = "No specific context found in knowledge base."
        else:
            context_texts = "\n\n".join([
                f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                for doc in retrieved_docs
            ])

        # Step 3: Generate response
        progress_bar.progress(75)
        status_text.text("🤖 Generating AI response with Groq...")

        # Format prompt
        prompt_template = load_prompt_template()
        formatted_prompt = prompt_template.format(
            combined_context=context_texts,
            user_question=user_question
        )

        # Call Groq API
        start_time = time.time()
        response = call_groq_llm(
            formatted_prompt,
            model=selected_model,
            max_tokens=max_tokens
        )
        end_time = time.time()

        # Complete progress
        progress_bar.progress(100)
        status_text.text("✅ Response generated!")

        # Clear progress after brief pause
        time.sleep(0.5)
        progress_container.empty()

        # Display results
        response_time = end_time - start_time

        # Show response
        st.markdown("### 🎯 Answer")
        st.markdown(response)

        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⚡ Response Time", f"{response_time:.2f}s")
        with col2:
            st.metric("📚 Sources Found", len(retrieved_docs))
        with col3:
            st.metric("🤖 Model", selected_model.split('-')[0].title())
        with col4:
            st.metric("🔧 Max Tokens", max_tokens)

        # Show source documents
        if retrieved_docs:
            with st.expander("📖 Source Documents Used"):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**📄 Document {i}: {doc.metadata.get('filename', 'Unknown')}**")
                    st.markdown(f"*{doc.metadata.get('char_count', 'N/A')} characters | {doc.metadata.get('word_count', 'N/A')} words*")
                    st.code(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                    if i < len(retrieved_docs):
                        st.markdown("---")

        # Debug information
        with st.expander("🔧 Technical Details"):
            st.json({
                "model_used": selected_model,
                "max_tokens": max_tokens,
                "response_time_seconds": round(response_time, 3),
                "documents_retrieved": len(retrieved_docs),
                "context_length": len(context_texts),
                "prompt_length": len(formatted_prompt)
            })

    except Exception as e:
        st.error(f"❌ Error processing your question: {str(e)}")
        logging.error(f"Error in main processing: {str(e)}")

        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common solutions:**
            1. **Check API key** - Ensure your Groq API key is valid
            2. **Try different model** - Some models may be temporarily unavailable
            3. **Reduce max tokens** - Lower the response length limit
            4. **Clear cache** - Use sidebar button to clear cached data
            5. **Check internet** - Ensure stable connection to Groq API
            """)

# Footer
st.markdown("---")
st.markdown(
    "🌱 **FarmAI** - Free AI-powered farming assistant | "
    "⚡ **Powered by [Groq](https://groq.com)** for lightning-fast inference | "
    "🏗️ **Built with** Streamlit • LangChain • FAISS"
)

# Deployment info
with st.expander("🚀 Free Deployment Guide"):
    st.markdown("""
    ### Deploy this app for free:

    **1. Streamlit Community Cloud** (Recommended):
    - Push code to GitHub
    - Connect at [share.streamlit.io](https://share.streamlit.io)
    - Add `GROQ_API_KEY` to secrets

    **2. Hugging Face Spaces**:
    - Create a Space with Streamlit
    - Add `GROQ_API_KEY` to Space secrets

    **3. Railway/Render**:
    - Deploy via GitHub integration
    - Set `GROQ_API_KEY` environment variable

    ### Files needed:
    ```
    ├── app.py (this file)
    ├── requirements.txt
    ├── farmai_training_data/ (your data)
    └── utils/text_data_preprocessing.py
    ```
    """)

    st.code("""
    # requirements.txt
    streamlit>=1.28.0
    langchain>=0.0.300
    langchain-community>=0.0.20
    langchain-huggingface>=0.0.1
    langchain-text-splitters>=0.0.1
    faiss-cpu>=1.7.4
    sentence-transformers>=2.2.2
    transformers>=4.21.0
    protobuf==3.20.3
    groq>=0.4.0
    """)
