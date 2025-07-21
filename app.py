from flask import Flask, request, jsonify
import requests
import os

# ✅ Use environment variable directly (Render sets this automatically)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Debug: Log incoming message and key snippet
    print("User message:", user_message)
    print("Loaded API key:", GROQ_API_KEY[:6] + "********")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for Verdefy, a sustainable fashion platform."},
            {"role": "user", "content": user_message}
        ],
        "model": "llama3-8b-8192",
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 1024,
        "stop": None,
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

# ✅ Entry point for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
