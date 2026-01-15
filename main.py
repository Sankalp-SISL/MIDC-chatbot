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
# CORS (REQUIRED FOR BROWSER)
# =====================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# =====================
# INTENT & FORM REGISTRY
# =====================

FORM_INTENT_REGISTRY = {
    "land rates": {
        "intent": "land_rates",
        "type": "external_form",
        "url": "https://land.midcindia.org/plotrate/index",
        "required_fields": [
            "Region",
            "Industrial Area",
            "Plot Type",
            "Plot Size"
        ],
        "confidence": 0.95
    },
    "invest with midc": {
        "intent": "investment_guidance",
        "type": "guided_navigation",
        "pages": [
            {
                "title": "Investors",
                "url": "https://www.midcindia.org/en/investors/"
            },
            {
                "title": "Contact MIDC",
                "url": "https://www.midcindia.org/en/contact/"
            }
        ],
        "confidence": 0.85
    }
}

# =====================
# LANGUAGE DETECTION
# =====================

def detect_language(text: str) -> str:
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x097F:
            return "mr"
    return "en"

# =====================
# GCS HELPERS
# =====================

def load_section(section: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{section}/content.json")

    if not blob.exists():
        return None

    return json.loads(blob.download_as_text())

def load_all_sections():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    documents = []
    for blob in blobs:
        if blob.name.endswith("content.json"):
            try:
                documents.append(json.loads(blob.download_as_text()))
            except Exception:
                continue
    return documents

# =====================
# INTENT DETECTION
# =====================

def detect_form_intent(question: str):
    q = question.lower()
    for key, meta in FORM_INTENT_REGISTRY.items():
        if key in q:
            return meta
    return None

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
    language_instruction = "Respond in Marathi." if language == "mr" else "Respond in English."

    # -------------------------------------------------
    # 1️⃣ FORM / AGENTIC INTENT (DETERMINISTIC)
    # -------------------------------------------------

    form_intent = detect_form_intent(question)

    if form_intent:
        response = {
            "answer": (
                "I can assist you with this request. "
                "This requires specific details to proceed."
            ),
            "confidence_score": form_intent.get("confidence", 0.8),
            "conversation_state": {
                "intent": form_intent["intent"],
                "last_interaction_ts": int(time.time()),
                "should_follow_up": True,
                "follow_up_message": (
                    "Please provide the required details so I can guide you further."
                )
            },
            "external_links": [],
            "forms_detected": [],
            "recommended_pages": []
        }

        if form_intent["type"] == "external_form":
            response["external_links"].append(form_intent["url"])
            response["forms_detected"].append({
                "form_name": "MIDC Official Form",
                "url": form_intent["url"],
                "required_fields": form_intent["required_fields"]
            })

        if form_intent["type"] == "guided_navigation":
            response["recommended_pages"] = form_intent["pages"]

        return jsonify(response)

    # -------------------------------------------------
    # 2️⃣ CONTENT-BASED ANSWERING (GCS + GEMINI)
    # -------------------------------------------------

    documents = load_all_sections()

    if not documents:
        return jsonify({
            "answer": "The knowledge base is currently unavailable.",
            "confidence_score": 0.0,
            "conversation_state": {"should_follow_up": False},
            "external_links": [],
            "forms_detected": [],
            "recommended_pages": []
        })

    context_chunks = []
    sources = []

    for doc in documents:
        chunks = doc.get("chunks", [])
        if chunks:
            context_chunks.extend(chunks[:2])
        if doc.get("source_url"):
            sources.append(doc["source_url"])

    context_text = "\n\n".join(context_chunks[:12])

    prompt = f"""
You are an official website assistant for MIDC.

CONTENT:
{context_text}

QUESTION:
{question}

RULES:
- Answer only from the content
- If the answer is partial, explain and suggest pages
- Do not hallucinate
- {language_instruction}
"""

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )

        answer_text = response.text.strip() if response and response.text else ""

        confidence = 0.7 if answer_text else 0.3

        return jsonify({
            "answer": answer_text or "The requested information is not available on MIDC’s official website.",
            "confidence_score": confidence,
            "conversation_state": {
                "intent": "general",
                "last_interaction_ts": int(time.time()),
                "should_follow_up": True,
                "follow_up_message": "Can I help you find a specific page or form?"
            },
            "external_links": list(set(sources))[:5],
            "forms_detected": [],
            "recommended_pages": [
                {"title": "MIDC Website", "url": "https://www.midcindia.org/"}
            ]
        })

    except Exception as e:
        return jsonify({
            "answer": "An internal error occurred while processing your request.",
            "confidence_score": 0.0,
            "conversation_state": {"should_follow_up": False},
            "error": str(e)
        }), 500

# =====================
# HEALTH CHECK
# =====================

@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot is running", 200

# =====================
# ENTRYPOINT
# =====================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
