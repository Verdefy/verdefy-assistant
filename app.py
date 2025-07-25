from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ✅ Load OpenAI key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# ✅ Flask app setup
app = Flask(__name__)
CORS(app)

# ✅ Load your trained Verdefy vector DB
DB_FOLDER = "verdefy_vector_db"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(DB_FOLDER, embeddings=embedding_model, allow_dangerous_deserialization=True)


# ✅ RetrievalQA chain that uses your custom data
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY),
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
