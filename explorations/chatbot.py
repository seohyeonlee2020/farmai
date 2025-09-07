import streamlit as st
import os
import json
import time
import requests
from text_data_preprocessing import *

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

@st.cache_data
def load_text_data():
    """Load text data from JSON file or extract from directory"""
    if not os.path.exists("./text_data.json"):
        directory = "./farmai_training_data"
        text_data = extract_text(directory)
        with open('text_data.json', 'w') as fp:
            json.dump(text_data, fp)
        return text_data
    else:
        with open('text_data.json') as f:
            text_data = json.load(f)
            return text_data

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
                "word_count": len(content.split())
            }
        )
        documents.append(doc)

    return documents

@st.cache_data
def prepare_documents():
    """Convert text data to chunked documents"""
    text_data = load_text_data()
    data = dict_to_documents(text_data)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_documents(data)

@st.cache_resource
def create_vectorstore():
    """Create and return vectorstore with embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=prepare_documents(),
        collection_name="farmai-embeddings",
        embedding=embeddings,
        persist_directory="./embeddings"
    )
    return vectorstore

@st.cache_data
def load_prompt_template():
    """Load prompt template from file"""
    with open('prompt_template.txt', 'r') as f:
        return f.read()

def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def call_ollama(prompt, model):
    """Send prompt to Ollama API"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }
        )
        return response.json()['response']
    except Exception as e:
        return f"Chatbot Error: {str(e)}"

# Streamlit App
st.title("FarmAI: Climate-Smart Farming Assistant")

# Check Ollama status
if not check_ollama_status():
    st.error("⚠️ Ollama is not running. Start it with 'ollama serve'")
    st.stop()

# Initialize vectorstore once per session
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base... This may take a moment."):
        st.session_state.vectorstore = create_vectorstore()
    st.success("✅ Knowledge base loaded successfully!")

# User input
user_question = st.text_input("Ask a farming question:")

if user_question:
    # Retrieve relevant documents
    retrieved_docs = st.session_state.vectorstore.similarity_search(user_question, k=2)

    # Show retrieval info
    st.info(f"Found {len(retrieved_docs)} relevant documents")

    # Create context from retrieved docs
    context_texts = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Load and format prompt
    prompt_template = load_prompt_template()
    prompt = prompt_template.format(context=context_texts, question=user_question)

    # Generate response
    model = 'qwen2:0.5b'

    with st.spinner("Generating response..."):
        start_time = time.process_time()
        llm_response = call_ollama(prompt, model)
        end_time = time.process_time()

    # Display results
    response_time = end_time - start_time
    st.write(f"**Response time:** {response_time:.2f} seconds")
    st.write("**Answer:**")
    st.write(llm_response)
