import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. LOAD THE CLEANED DATA ---
print("Loading cleaned data...")
loader = TextLoader("cleaned_llm_data.txt", encoding="utf-8")
documents = loader.load()

# --- 2. CHUNK THE TEXT ---
print("Chunking text for embeddings...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  
    chunk_overlap=50 
)
chunks = text_splitter.split_documents(documents)
print(f"Successfully split data into {len(chunks)} chunks.")

# --- 3. GENERATE OPEN-SOURCE EMBEDDINGS & STORE IN FAISS ---
print("Downloading open-source embedding model (this only takes a moment)...")
# We use a highly efficient, free open-source model from HuggingFace
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building FAISS Vector DB...")
vector_db = FAISS.from_documents(chunks, embeddings_model)

# --- 4. SAVE THE DATABASE LOCALLY ---
vector_db.save_local("faiss_audit_index")
print("Success! FAISS Vector Database built and saved to the 'faiss_audit_index' folder.")