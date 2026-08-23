# chatbot_engine.py
import os
from dotenv import load_dotenv
import time
import json
import logging
import requests
import warnings
import faulthandler

faulthandler.enable()
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.text_data_preprocessing import extract_text

# Default Paths (Override as needed)
TRAIN_DIR = os.getenv("TRAIN_DIR")
TEXT_JSON = "./disaster_relief_text_data.json"
INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_PATH = "./hf-embeddings"


def load_text_data(train_dir=TRAIN_DIR, text_filename=TEXT_JSON):
    """Load text data from JSON file or extract from directory."""
    try:
        if not os.path.exists(text_filename):
            logging.info("JSON file not found, extracting text from training dir...")
            text_data = extract_text(train_dir)
            with open(text_filename, "w", encoding="utf-8") as fp:
                json.dump(text_data, fp, ensure_ascii=False, indent=4)
            return text_data
        else:
            with open(text_filename, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error in load_text_data: {str(e)}")
        return {}


def dict_to_documents(file_dict):
    """Convert file dictionary to LangChain Document objects."""
    documents = []
    if not file_dict:
        return documents

    for filename, content in file_dict.items():
        if not content or not str(content).strip():
            continue
        doc = Document(
            page_content=str(content).strip(),
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


def prepare_documents():
    """Convert text data to chunked documents"""
    text_data = load_text_data()
    if not text_data:
        logging.warning("No text data available for processing")
        return []

    data = dict_to_documents(text_data)
    if not data:
        logging.warning("No valid documents created from text data")
        return []

        # Use smaller chunks for better retrieval with overlapping
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increased chunk size for better context
        chunk_overlap=50,  # Increased overlap for continuity
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True,
        length_function=len,
    )

    split_docs = text_splitter.split_documents(data)
    logging.info(f"Split {len(data)} documents into {len(split_docs)} chunks")
    return split_docs


def create_embeddings():
    try:
        # Initialize and save embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_PATH,
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logging.info(f"Creating embeddings with model {EMBEDDING_MODEL_NAME}")
        return embeddings

    except Exception as e:
        error_msg = f"Error creating embeddings: {str(e)}"
        #  st.error(error_msg)
        logging.error(error_msg)
        raise


def create_vectorstore(index_path=INDEX_PATH, embeddings=None):
    if not embeddings:
        embeddings = create_embeddings()

    # if FAISS index already exists, load it
    faiss_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_path) and os.path.isfile(faiss_file):
        print("Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,  # Required for loading pickled data
        )
    else:
        # else, create vectorstore from documents and embeddings
        documents = prepare_documents()
        logging.info(f"Creating FAISS vectorstore with {len(documents)} documents")

        # Create FAISS vectorstore with error handling
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

        # Persist FAISS
        vectorstore.save_local(index_path)
        print(f"Index saved to {index_path}")
    logging.info("Successfully created FAISS vectorstore")
    return vectorstore


def load_prompt_template():
    """Load prompt template from file with fallback"""
    template_path = "utils/prompt_template.txt"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        logging.warning(f"Prompt template file not found: {template_path}")
        # Return default template
        return """Based on the following knowledge context, provide a helpful and accurate answer to the user's question.
                Context: {combined_context}
                Question: {user_question}

                Answer: Provide a clear, practical answer based on the context above.
                If the context doesn't contain relevant information, acknowledge this and provide general guidance where appropriate."""


def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
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


def get_model_response(prompt, answering_model):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": answering_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": 512,
                },
                "keep_alive": 0,
            },
            timeout=60,  # Increased timeout for model processing
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            return f"Ollama API Error: HTTP {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. The model might be processing a complex query."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please ensure Ollama is running."
    except Exception as e:
        return f"Chatbot Error: {str(e)}"


class ChatbotEngine:
    def __init__(self, index_path=INDEX_PATH):
        self.documents = prepare_documents()
        self.embeddings = create_embeddings()
        self.vectorstore = create_vectorstore(
            index_path=INDEX_PATH, embeddings=self.embeddings
        )

    # TODO: incorporate model choice
    def generate_model_context(
        self, k: int = 2, user_question: str = "heatwave safety tips"
    ) -> str:
        # 1. Similarity search
        retrieved_docs = self.vectorstore.similarity_search(user_question, k=k)
        logging.info(f"Retrieved {len(retrieved_docs)} documents")

        if not retrieved_docs:
            logging.warning("No documents retrieved for user question")

        else:
            # Create context from retrieved docs
            context_from_retrieved_docs = "\n\n".join(
                [
                    f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                    for doc in context_from_retrieved_docs
                ]
            )
            print("context being added", context_from_retrieved_docs)
            return context_from_retrieved_docs

    def dynamically_generate_context(context_from_retrieved_docs, user_question: str = "heatwave safety tips")
            # load and format prompt
            prompt_template = load_prompt_template()
            prompt = prompt_template.format(
                combined_context=context_from_retrieved_docs, user_question=user_question
            )
            # return dynamically combined prompt
            return prompt

    def get_model_response(prompt, answering_model):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": answering_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "temperature": 0.4,
                        "top_p": 0.9,
                        "max_tokens": 512,
                    },
                    "keep_alive": 0,
                },
                timeout=60,  # Increased timeout for model processing
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return (
                    f"Ollama API Error: HTTP {response.status_code} - {response.text}"
                )

        except requests.exceptions.Timeout:
            return "Request timed out. The model might be processing a complex query."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Please ensure Ollama is running."
        except Exception as e:
            return f"Chatbot Error: {str(e)}"


"""
    def query(
        self, user_question: str, k: int = 2, answering_model="qwen2:0.5b"
    ) -> str:
        #Pipeline execution method called during inference or evaluation.
        # 1. Similarity search
        retrieved_docs = self.vectorstore.similarity_search(user_question, k=k)
        logging.info(f"Retrieved {len(retrieved_docs)} documents")

        if not retrieved_docs:
            logging.warning("No documents retrieved for user question")
        else:
            # Create context from retrieved docs
            context_texts = "\n\n".join(
                [
                    f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                    for doc in retrieved_docs
                ]
            )
            print("context being added", context_texts)

            # load and format prompt
            prompt_template = load_prompt_template()
            prompt = prompt_template.format(
                combined_context=context_texts, user_question=user_question
            )

            # generate response
            response_start_time = time.time()

            model_response = get_model_response(prompt, answering_model)

            response_end_time = time.time()
            logging.info(f"model response:{model_response}")
        return model_response
"""

engine = ChatbotEngine()
engine.query("heatwave safety tips")
