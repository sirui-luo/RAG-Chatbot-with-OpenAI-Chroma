# build_index.py
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load env variables
load_dotenv("api.env")

CHROMA_PATH = "chroma/"

# Remove existing Chroma DB if exists
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Load markdown docs
def load_documents():
    loader = DirectoryLoader("data/books/", glob="*.md")
    return loader.load()

# Split documents
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

# Load docs
docs = load_documents()
chunks = chunk_documents(docs)
print(f"✅ Loaded {len(docs)} documents, split into {len(chunks)} chunks")

# Init embedding
embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Save to Chroma
db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory=CHROMA_PATH,
    collection_name="alice_books"
)

print("✅ Chroma DB built and saved to disk.")
