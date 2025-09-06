
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import json
import requests
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline

from utils.text_data_preprocessing import *
import os

load_dotenv()
GROQ_API_KEY= os.getenv('GROQ_API_KEY')

#dump text_data as json if not already done
if not os.path.exists("./text_data.json"):
	directory = "./farmai_training_data"
	text_data =  extract_text(directory)
	with open('text_data.json', 'w') as fp:
	    json.dump(text_data, fp)
else:
	with open('text_data.json') as f:
	    text_data = json.load(f)
	    print(text_data)

# Load a plain text document
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
                "filename": os.path.basename(filename),  # Just the filename without path
                "file_extension": os.path.splitext(filename)[1],
                "char_count": len(content),
                "word_count": len(content.split())
            }
        )
        documents.append(doc)

    return documents

# Split into manageable chunks
data = dict_to_documents(text_data)
# text to chunks
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=250,
chunk_overlap=20,
separators=["\n\n", "\n", ". ", " ", ""],
keep_separator=True)
all_splits = text_splitter.split_documents(data)


#TODO: switch to model that performs better on manuals. MiniLM is for rapid prototyping
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#create persistent client in chromadb
chroma_client = chromadb.PersistentClient(path="./embeddings")
# Add to Vector database ChromaDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma-2",
    embedding=embeddings,
)

#connect to groq mixtral
from langchain_groq import ChatGroq

# Instantiate the Groq model (use your API key)
model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768"
)

from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)
