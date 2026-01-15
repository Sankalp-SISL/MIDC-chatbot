import os
import json
import time
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

app = Flask(__name__)

# =====================
# CORS
# =====================

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# =====================
# LANGUAGE DETECTION
# =====================

def detect_language(text: str) -> str:
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x097F:
            return "mr"
    return "en"

# =====================
# INTERNET TRIGGER
# =====================

def is_internet_query(question: str, mode: str | None):
    if mode == "internet":
        return True

    triggers = [
        "search internet",
        "outside midc",
        "general knowledge",
        "google",
        "latest news",
        "not on midc"
    ]
    return any(t in question.lower() for t in triggers)

# =====================
# GCS LOADING
# =====================

def load_all_content():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    pages, pdfs, forms, links = [], [], [], []

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        data = json.loads(blob.download_as_text())
        meta = data.get("metadata", {})

        if meta.get("type") == "pdf":
            pdfs.append(data)
        elif meta.get("type") == "form":
            forms.append(data)
        elif meta.get("type") == "external":
            links.append(data)
        else:
            pages.append(data)

    return pages, pdfs, forms, links

# =====================
# CONTEXT BUILDER
# =====================

def build_context(pages, pdfs):
    chunks = []

    for p in pages[:5]:
        chunks.extend(p.get("chunks", [])[:2])

    for p in pdfs[:3]:
        chunks.extend(p.get("chunks", [])[:2])

    return "\n\n".join(chunks)

# =====================
# INTERNET ANSWER
# =====================

def internet_answer(question, language_instruction):
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(
                    text=f"""
Use Google Search to answer this question.
Clearly state this is NOT official MIDC information.
{language_instruction}

QUESTION:
{question}
"""
                )]
            )
        ],
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )

    return response.text if response and response.text else None

# =====================
# MAIN CHAT ROUTE
# =====================

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json() or {}
    question = body.get("question", "").strip()
    mode = body.get("mode")

    if not question:
        return jsonify({"error": "Question required"}), 400

    language = detect_language(question)
    language_instruction = "Respond in Marathi." if language == "mr" else "Respond in English."

    # 🔹 INTERNET MODE
    if is_internet_query(question, mode):
        answer = internet_answer(question, language_instruction)
        return jsonify({
            "answer": answer,
            "confidence_score": 0.65,
            "sources": ["Google Search"],
            "external_links": [],
            "forms_detected": [],
            "recommended_pages": [],
            "conversation_state": {
                "intent": "internet",
                "should_follow_up": False
            }
        })

    # 🔹 MIDC MODE
    pages, pdfs, forms, links = load_all_content()
    context = build_context(pages, pdfs)

    prompt = f"""
You are an official AI assistant for MIDC.

CONTENT:
{context}

QUESTION:
{question}

RULES:
- Answer ONLY from MIDC content
- If forms or external links exist, guide the user
- Provide clickable links
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer = response.text if response and response.text else "Information not available."

    return jsonify({
        "answer": answer,
        "confidence_score": 0.82 if answer else 0.0,
        "sources": list({p.get("source_url") for p in pages if p.get("source_url")}),
        "external_links": [l.get("url") for l in links[:3]],
        "forms_detected": forms[:2],
        "recommended_pages": [
            {"title": p.get("title"), "url": p.get("source_url")}
            for p in pages[:3]
        ],
        "conversation_state": {
            "intent": "midc",
            "should_follow_up": True,
            "follow_up_message": "Would you like help navigating a page or filling a form?"
        }
    })

@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot Backend Running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
