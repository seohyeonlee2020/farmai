# streamlit wrapper
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOBUF_PYTHON_IMPLEMENTATION"] = "python"

# Force Python to prioritize local root directory
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

import streamlit as st
import json
import time
import requests
import logging
import warnings
import faulthandler

faulthandler.enable()
# Suppress protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=FutureWarning)

from chatbot_engine import ChatbotEngine
from chatbot_engine import *

# Streamlit App Configuration
st.set_page_config(
    page_title="Advice Delivered Offline",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main App
# TODO: Change title
st.title("Offline Disaster Relief")
st.title("sLM + RAG")

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

    # TODO: figure out ways to incorporate model choice
    user_model_choice = st.selectbox(
        "Model:", ["qwen2:0.5b", "qwen3:0.6b", "tinyllama"], index=0
    )


# Cache engine instance so FAISS index loads into memory only ONCE per session
@st.cache_resource
def load_engine():
    with st.spinner("🔄 Loading knowledge base... This may take a moment."):
        return ChatbotEngine()


try:
    engine = load_engine()
    st.success("✅ Knowledge base loaded successfully!")
# TODO: figure out what exception to catch
except Exception as e:
    st.error(f"Failed to initialize engine: {e}")
    st.stop()

# chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# take question from user
if user_question := st.chat_input(
    placeholder="Heatwave safety tips?",
):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not ollama_running:
            st.warning("⚠️ Ollama is not running. Please start Ollama to get responses.")
        else:
            try:
                logging.info(f"Processing user question: {user_question}")
                # Create progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Retrieve relevant documents
                status_text.text("🔍 Searching for relevant documents...")
                progress_bar.progress(25)
                output = engine.retrieve_relevant_context(user_question)
                logging.info(f"output retrieval: {output}")
                context_from_retrieved_docs = output["output"]
                retrieval_success = output["retrieval_success"]
                retrieval_time = output["retrieval_time"]

                if not retrieval_success:
                    st.warning(
                        "❓ No relevant documents found. Try rephrasing your question or using more specific terms."
                    )
                    logging.warning("No documents retrieved for user question")
                else:
                    # Step 2: combine retrieved docs, user question, and prompt template to generate context given to sLM
                    prompt = engine.dynamically_generate_context(
                        context_from_retrieved_docs, user_question
                    )

                    # Step 3: Generate response
                    progress_bar.progress(75)
                    status_text.text("🤖 Generating AI response...")

                    output = engine.get_model_response(
                        prompt, answering_model=user_model_choice
                    )

                    model_response = output["output"]
                    response_time = output["response_time"]

                    # model_response = engine.query(user_question, answering_model=user_model_choice)

                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("✅ Response generated!")
                    logging.info(f"Total response time: {response_time:2f} seconds")

                    # log response
                    logging.info(f"response: {model_response}")

                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # Display results
                # response_time = response_end_time - response_start_time

                # Show response
                st.markdown("### 🎯 Answer")
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(model_response)

                # Save AI response to persistent state
                st.session_state.messages.append(
                    {"role": "assistant", "content": model_response}
                )

            except Exception as e:
                error_msg = f"Error processing your question: {str(e)}"
                logging.error(error_msg)
                st.error(error_msg)
