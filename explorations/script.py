# %%
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


# %%
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


# %%

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


# %%
# text to chunks
data = dict_to_documents(text_data)
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=250,
chunk_overlap=20,
separators=["\n\n", "\n", ". ", " ", ""],
keep_separator=True)
all_splits = text_splitter.split_documents(data)


# %%

#TODO: switch to model that performs better on manuals. MiniLM is for rapid prototyping
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# %%
import chromadb


chroma_client = chromadb.PersistentClient(path="./embeddings")

# Add to Vector database ChromaDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="farmai-embeddings",
    embedding=embeddings,
	persist_directory="./embeddings" #needed to use persistent client
)


# %%
# check vector db response.
# this will become a context to LLM. LLM has to respond from this context
question = 'Which crops to plant after harvesting maize?'
docs = vectorstore.similarity_search(question)


# %%
import re
len(docs)
print(docs[1])


# %%
#take user input (cli)
user_question = input("Ask a question")
retrieved_docs = vectorstore.similarity_search(question, k=2)


# %%

print(f"Found {len(retrieved_docs)} relevant documents")

# STEP 2: Extract and combine context
print("\nStep 2: Extracting context...")
context_texts = "".join(doc.page_content for doc in retrieved_docs)
combined_context = "\n\n".join(context_texts)


# %%
print("\nStep 3: Creating prompt...")
prompt = f"""Context Information:
{combined_context}

User Question: {user_question}

Instructions: You are providing climate-smart farming advice to women smallholder farmers in Uganda.
Based ONLY on the context provided above, create a very concise 1-2 sentence answer that is:

- SIMPLE & ACTIONABLE: Give specific steps using "You can..." format
- LOCALLY RELEVANT: Reference materials available in Uganda (banana leaves, cassava peels, local wood, etc.)
- TRUSTWORTHY: Reference sources if mentioned in context
- INCLUSIVE: Focus on low-cost practices accessible to smallholders
- CLIMATE-FOCUSED: Address weather challenges

If the context doesn't contain relevant farming advice, respond: "I don't have enough information to provide safe farming advice for that question."
Think step by step.
Answer:"""


# %%

def call_ollama(prompt, model):
    """Send prompt to Ollama API"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,  # or whatever model you installed
                'prompt': prompt,
                'stream': False
            }
        )
        return response.json()['response']
    except Exception as e:
        return f"LLM Error: {str(e)}"


# %%
# STEP 4: Send to LLM (Example with Ollama)
print("\nStep 4: Getting response...")
model = 'tinyllama'
llm_response = call_ollama(prompt, model)

print(f"\nFinal Answer: {llm_response}")


# %% [markdown]
# zero-shot RAG prompt
#
# Context Information:
# {retrieved_context}
#
# User Question: {user_query}
#
# Instructions: You are providing climate-smart farming advice to women smallholder farmers in Uganda. Based ONLY on the context provided above, create a concise 1-2 sentence answer that is:
#
# - SIMPLE & ACTIONABLE: Give specific steps using "You can..." format with clear actions
# - LOCALLY RELEVANT: Reference materials commonly available in Uganda (banana leaves, cassava peels, local wood, clay, ash, etc.)
# - TRUSTWORTHY: If the context mentions sources like extension services, cooperatives, or farmer experiences, reference them
# - INCLUSIVE: Focus on low-cost or free practices accessible to smallholders
# - CLIMATE-FOCUSED: Address weather challenges like irregular rains, drought, or flooding
#
# If the context doesn't contain relevant farming advice for the specific question, respond: "I don't have enough information to advise on that your question."
#
# Answer:


