# Production-Grade GenAI Assistant with RAG
This project is a GenAI assistant using Retrieval-Augmented Generation (RAG).
## Features
- Document question answering
- Semantic search
- LLM generated responses

## Tech Stack
- Python
- Flask
- python dotenv
- Vector embeddings
- Claude API
  

## 🏗 Architecture Diagram (Text Version)
    ┌─────────────┐
    │   Frontend  │  HTML Chat Interface
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │   Backend   │  Flask API (/api/chat)
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ Vector Store│  Embeddings JSON
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │    RAG      │  Retrieval-Augmented Generation
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │     LLM     │  Claude API
    └─────────────┘

    
---

## 📊 RAG Workflow Explanation (Text Version)
User Question
│
▼
Generate Query Embedding (Sentence Transformers)
│
▼
Similarity Search in Vector Store
│
▼
Retrieve Top 3 Document Chunks
│
▼
Construct Prompt with Context + Conversation History
│
▼
Send Prompt to LLM (Claude)
│
▼
Generate Grounded Response
│
▼
Return JSON Response via API

**Key Points:**

- Ensures factual answers using document context  
- Maintains conversation history for multi-turn interactions  
- Uses similarity threshold to avoid hallucinations

---

## ⚡ Embedding Strategy

- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)  
- **Chunking**: Split documents into 300–500 token chunks  
- **Vector Storage**: JSON (`vector_store.json`)  
- **Purpose**: Represent document meaning in high-dimensional space for semantic search

---

## 🔍 Similarity Search Explanation

- Uses **cosine similarity** between query embedding and document chunk embeddings:

\[
\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}
\]

- Retrieves **top 3 most relevant chunks**  
- Similarity threshold ensures fallback response when context is insufficient  
- Retrieval happens **before LLM call** to ground responses

---

## ✏️ Prompt Design Reasoning

Prompt is designed to:

1. Prioritize **document context** over conversation history  
2. Include last **3–5 message pairs**  
3. Instruct model to respond `"I don't have enough information"` if context is missing  
4. Control temperature for deterministic outputs (0–0.3)  
5. Reduce hallucinations and ensure accurate answers

**Example Prompt:**
You are a helpful assistant.

Use ONLY the provided context.

Context:
{retrieved_document_chunks}

Conversation History:
User: Hi
Assistant: Hello!
User: How do I reset my password?

Question:
How can I reset my password?

If the answer is not in the context say:
"I don't have enough information."

---

## Run the Project

pip install -r requirements.txt
python build_index.py
python app.py
