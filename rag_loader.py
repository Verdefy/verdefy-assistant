from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# Load all .txt files from verdefy_knowledge folder
docs = []
knowledge_dir = "verdefy_knowledge"
for filename in os.listdir(knowledge_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(knowledge_dir, filename), encoding="utf-8")
        docs.extend(loader.load())

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Embed with HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector DB
db = FAISS.from_documents(split_docs, embedding_model)

# Save vector DB to local folder
db.save_local("verdefy_vector_db")

print("✅ Vector DB created and saved to verdefy_vector_db/")
