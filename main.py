import os
import json
from flask import Flask, request, jsonify
from google.cloud import storage
from google import genai
from google.genai import types

# =====================
# CONFIGURATION
# =====================

BUCKET_NAME = "midc-general-chatbot-bucket-web-data"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

app = Flask(__name__)

# =====================
# CORS
# =====================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response

# =====================
# LANGUAGE DETECTION
# =====================

def detect_language(text: str) -> str:
    return "mr" if any(0x0900 <= ord(c) <= 0x097F for c in text) else "en"

# =====================
# SAFE GCS LOADING
# =====================

def load_relevant_documents(limit=15):
    """
    Loads a limited number of documents safely.
    """
    docs = []
    try:
        for blob in bucket.list_blobs():
            if not blob.name.endswith("content.json"):
                continue
            docs.append(json.loads(blob.download_as_text()))
            if len(docs) >= limit:
                break
    except Exception:
        pass
    return docs

# =====================
# SEMANTIC FILTERING
# =====================

def semantic_filter(question, docs):
    """
    Uses Gemini to select relevant documents.
    """
    index_map = {i: d for i, d in enumerate(docs)}

    summary = "\n".join(
        f"[{i}] {d.get('section')} - {d.get('source_url')}"
        for i, d in index_map.items()
    )

    prompt = f"""
From the document list below, choose the most relevant ones
to answer the question.

Return JSON array of indices.

Documents:
{summary}

Question:
{question}
"""

    try:
        r = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            generation_config={"temperature": 0}
        )
        idxs = json.loads(r.text.strip())
        return [index_map[i] for i in idxs if i in index_map]
    except Exception:
        return []

# =====================
# CHAT ROUTE
# =====================

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    language = detect_language(question)

    docs = load_relevant_documents()
    matched_docs = semantic_filter(question, docs)

    context = []
    pages = []
    forms = []
    links = []

    for d in matched_docs:
        context.extend(d.get("chunks", [])[:2])
        if d.get("source_url"):
            pages.append({
                "title": d.get("section", "").replace("-", " ").title(),
                "url": d["source_url"]
            })
        forms.extend(d.get("forms", []))
        links.extend(d.get("related_links", []))

    prompt = f"""
You are an official MIDC website assistant.

Use ONLY the information provided below.
If a page or form exists, guide the user to it.

Context:
{chr(10).join(context)}

Question:
{question}

Respond in {"Marathi" if language == "mr" else "English"}.
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    follow_up = None
    if forms:
        follow_up = "I can help you fill the required form step-by-step. Would you like to proceed?"
    elif pages:
        follow_up = "Would you like me to open the relevant page or guide you further?"

    return jsonify({
        "answer": response.text.strip(),
        "recommended_pages": pages,
        "forms_detected": forms,
        "external_links": links,
        "confidence_score": round(min(1.0, len(context) / 8), 2),
        "conversation_state": {
            "should_follow_up": bool(follow_up),
            "follow_up_message": follow_up
        }
    })

@app.route("/", methods=["GET"])
def health():
    return "MIDC Agentic Assistant running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
