import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import os

load_dotenv()

app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("vector_store.json") as f:
    vector_store = json.load(f)

sessions = {}


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_chunks(query):

    query_embedding = model.encode(query)

    scores = []

    for doc in vector_store:

        sim = cosine_similarity(query_embedding, doc["embedding"])

        scores.append((sim, doc))

    scores.sort(reverse=True)

    top = scores[:3]

    chunks = [d["content"] for s, d in top]

    similarities = [s for s, d in top]

    print("Similarity scores:", similarities)

    return chunks


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.json

    if "sessionId" not in data or "message" not in data:
        return jsonify({"error": "Invalid request"}), 400

    session_id = data["sessionId"]
    message = data["message"]

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id][-6:]

    chunks = retrieve_chunks(message)

    context = "\n".join(chunks)

    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context.

Context:
{context}

Conversation History:
{history}

Question:
{message}

If the answer is not in the context say:
"I don't have enough information."
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    reply = response.content[0].text

    sessions[session_id].append({
        "user": message,
        "assistant": reply
    })

    return jsonify({
        "reply": reply,
        "retrievedChunks": len(chunks),
        "tokensUsed": response.usage.output_tokens
    })


if __name__ == "__main__":
    app.run(debug=True)