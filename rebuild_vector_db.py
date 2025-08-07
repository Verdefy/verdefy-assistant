import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ✅ Correct import

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")

print("✅ Google API Key Loaded Successfully")

# Load all .txt files from your knowledge folder
loader = DirectoryLoader(
    "verdefy_knowledge",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}  # Avoids UnicodeDecodeError
)

documents = loader.load()
print(f"✅ Loaded {len(documents)} documents from verdefy_knowledge")

# Create embeddings using Gemini
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Build FAISS Vector DB
db = FAISS.from_documents(documents, embedding_model)
db.save_local("verdefy_vector_db")

print("✅ Vector DB built and saved to verdefy_vector_db")
