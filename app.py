from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Add this line
import requests
import os

# ✅ Load environment variable for GROQ
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable")

app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin requests (from your Elementor frontend)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Debug logs
    print("User message:", user_message)
    print("Loaded API key:", GROQ_API_KEY[:6] + "********")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

payload = {
  "messages": [
    {
      "role": "system",
      "content": """You are Verda, a helpful assistant for Verdefy — Nigeria’s online thrift marketplace. 
You ONLY talk about thrift, not sustainable fashion. Verdefy helps Nigerians easily buy and sell authentic second-hand clothes.
Always respond in a helpful, succinct, friendly tone. Assume you’re based in Nigeria. Guide users on tracking orders, browsing, or selling their clothes."""
    },
    {"role": "user", "content": user_message}
  ],
  "model": "llama3-8b-8192",
  "temperature": 0.7,
  "top_p": 1,
  "max_tokens": 1024,
  "stream": False
}

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Groq error response:", response.text)
        return jsonify({"error": "Something went wrong", "details": response.text}), 500

    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    return jsonify({"response": reply})

# ✅ Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
