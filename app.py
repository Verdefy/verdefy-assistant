from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# LangChain bits
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ===== Load env =====
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# ===== Flask =====
app = Flask(__name__)
CORS(app)

# ===== Load your FAISS vector DB =====
DB_FOLDER = "verdefy_vector_db"
# Embeddings must match what you used to build the DB
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

db = FAISS.load_local(
    DB_FOLDER,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ===== Retrieval chain (OpenAI LLM) =====
retriever = db.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        answer = qa_chain.run(user_message)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
