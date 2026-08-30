import streamlit as st
import os
import json
import time
import requests
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


@st.cache_data
def load_text_data():
    with open("text_data.json", "r") as f:
        print("opening JSON file")
        json_data = json.load(f)
        return json_data


# Initial version focused on farming advice for low-resource communities, especially women farmers with limited land rights


def dict_to_documents(file_dict):
    """
    Convert a dictionary of {filename: content} to LangChain Document objects

    Args:
        file_dict (dict): Dictionary with filename as key, text content as value

    Returns:
        list: List of Document objects
    """
    documents = []

    for filename, content in file_dict.items():
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "filename": os.path.basename(filename),
                "file_extension": os.path.splitext(filename)[1],
                "char_count": len(content),
                "word_count": len(content.split()),
            },
        )
        documents.append(doc)

    return documents


@st.cache_data
def prepare_documents():
    """Convert text data to chunked documents"""
    print("chunking text data")
    training_data = load_text_data()
    data = dict_to_documents(training_data)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return text_splitter.split_documents(data)


@st.cache_resource
def create_vectorstore():
    """Create and return vectorstore with embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=prepare_documents(),
        collection_name="agriadvice-embeddings",
        embedding=embeddings,
        persist_directory="./embeddings",
    )
    return vectorstore


def load_vectorstore():
    # Initialize vectorstore once per session
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading knowledge base... This may take a moment."):
            st.session_state.vectorstore = create_vectorstore()
        st.success("✅ Knowledge base loaded successfully!")


@st.cache_data
def load_prompt_template():
    """Load prompt template from file"""
    with open("utils/prompt_template.txt", "r") as f:
        return f.read()


def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def call_ollama(prompt, model):
    """Send prompt to Ollama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        return response.json()["response"]
    except Exception as e:
        return f"Chatbot Error: {str(e)}"


# Streamlit App
st.title("Offline Chatbot")

# Check Ollama status
if not check_ollama_status():
    st.error("⚠️ Ollama is not running. Start it with 'ollama serve'")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a farming question:"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            st.info("Loading...")
            load_vectorstore()
            logging.info(f"Processing user question: {user_question}")
            # Retrieve relevant documents
            logging.info("Searching for relevant documents...")

            retrieved_docs = st.session_state.vectorstore.similarity_search(
                user_question, k=2
            )
            logging.info(f"Retrieved {len(retrieved_docs)} documents")

            # Show retrieval info
            st.info(f"Found {len(retrieved_docs)} relevant documents")

            if not retrieved_docs:
                st.warning("No relevant documents found. Try rephrasing your question.")
                logging.warning("No documents retrieved for user question")
            else:
                # Create context from retrieved docs
                logging.info("Creating context from retrieved documents...")
                context_texts = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # Load and format prompt
                logging.info("Loading and formatting prompt...")
                prompt_template = load_prompt_template()
                prompt = prompt_template.format(
                    combined_context=context_texts, user_question=user_question
                )

                # Generate response
                model = "qwen2:0.5b"
                logging.info(f"Generating response with model: {model}")

                with st.spinner("Generating response..."):
                    start_time = time.process_time()
                    response = call_ollama(prompt, model)
                    end_time = time.process_time()
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                # Display results
                response_time = end_time - start_time
                logging.info(f"Total response time: {response_time:.2f} seconds")

                st.write(f"**Response time:** {response_time:.2f} seconds")
                st.write("**Answer:**")
                st.write(response)

        except Exception as e:
            logging.error(f"Error processing user question: {str(e)}")
            st.error(f"An error occurred while processing your question: {str(e)}")
            st.info("Please check the logs for more details.")

        # Show debug info in expander
        with st.expander("🔍 Debug Information"):
            st.write(f"**Retrieved {len(retrieved_docs)} documents:**")
            for i, doc in enumerate(retrieved_docs):
                st.write(
                    f"**Document {i + 1}:** {doc.metadata.get('filename', 'Unknown')}"
                )
                st.write(f"*Content preview:* {doc.page_content[:200]}...")
