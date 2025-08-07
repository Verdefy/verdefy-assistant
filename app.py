from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Load vector DB
DB_FOLDER = "verdefy_vector_db"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
db = FAISS.load_local(DB_FOLDER, embeddings=embedding_model, allow_dangerous_deserialization=True)

# ✅ Set up Gemini-based RetrievalQA
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = qa_chain.run(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
