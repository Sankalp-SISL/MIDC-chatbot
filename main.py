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
# 🔒 MIDC ENTITY SAFETY LOCK (NEW)
# =====================

MIDC_ENTITY_KEYWORDS = [
    "ceo",
    "managing director",
    "md",
    "chairman",
    "contact",
    "email",
    "phone",
    "officer",
    "official",
    "midc head"
]

def is_midc_entity_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in MIDC_ENTITY_KEYWORDS)

# =====================
# INTERNET MODE TRIGGER (UNCHANGED)
# =====================

def is_internet_query(question: str, mode: str | None):
    if mode == "internet":
        return True

    triggers = [
        "search internet",
        "search web",
        "outside midc",
        "general knowledge",
        "latest news",
        "google"
    ]
    return any(t in question.lower() for t in triggers)

# =====================
# LOAD ALL GCS CONTENT
# =====================

def load_all_content():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    pages, pdfs, forms, external_links = [], [], [], []

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        try:
            data = json.loads(blob.download_as_text())
            meta = data.get("metadata", {})

            content_type = meta.get("type", "page")

            if content_type == "pdf":
                pdfs.append(data)
            elif content_type == "form":
                forms.append(data)
            elif content_type == "external":
                external_links.append(data)
            else:
                pages.append(data)

        except Exception:
            continue

    return pages, pdfs, forms, external_links

# =====================
# CONTEXT BUILDER (BOOSTED)
# =====================

def build_context(pages, pdfs, force_leadership=False):
    context_chunks = []

    # 🔹 Force About / Contact pages for leadership queries
    if force_leadership:
        for p in pages:
            title = (p.get("title") or "").lower()
            if any(k in title for k in ["about", "contact", "midc"]):
                context_chunks.extend(p.get("chunks", [])[:4])

    for p in pages[:8]:
        context_chunks.extend(p.get("chunks", [])[:2])

    for p in pdfs[:5]:
        context_chunks.extend(p.get("chunks", [])[:2])

    return "\n\n".join(context_chunks)

# =====================
# INTERNET ANSWER (CONTROLLED – UNCHANGED)
# =====================

def internet_answer(question, language_instruction):
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(
                    text=f"""
You are an assistant using Google Search.

IMPORTANT:
- This information is NOT official MIDC data
- Do NOT hallucinate
- Use ONLY verifiable web results
- Respond in valid HTML ONLY
- No markdown
- No '*' or '**'
- Use <strong>, <ul>, <li>, <a>

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
    language_instruction = (
        "Respond in Marathi." if language == "mr"
        else "Respond in English."
    )

    # =====================
    # 🔒 MIDC ENTITY ALWAYS WINS (NEW)
    # =====================

    force_midc = is_midc_entity_query(question)

    # =====================
    # INTERNET MODE (GUARDED)
    # =====================

    if not force_midc and is_internet_query(question, mode):
        answer = internet_answer(question, language_instruction)

        return jsonify({
            "answer": answer,
            "confidence_score": 0.65,
            "recommended_pages": [],
            "external_links": [],
            "forms_detected": [],
            "conversation_state": {
                "intent": "internet",
                "should_follow_up": False
            }
        })

    # =====================
    # MIDC MODE (PRIMARY)
    # =====================

    pages, pdfs, forms, external_links = load_all_content()
    context = build_context(pages, pdfs, force_leadership=force_midc)

    prompt = f"""
You are an official AI assistant for
Maharashtra Industrial Development Corporation (MIDC).

CONTENT:
{context}

QUESTION:
{question}

MANDATORY FORMAT RULES:
- OUTPUT VALID HTML ONLY
- NO markdown
- NO '*' or '**'
- Use <strong> for headings
- Use <ul><li> for lists
- Use <a href=""> for links
- Clean spacing with <br>
- If information exists, answer confidently
- If external site exists (land rates, tenders), mention and link it
- Do NOT hallucinate
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer = (
        response.text.strip()
        if response and response.text
        else "<strong>Information not available on the MIDC website.</strong>"
    )

    return jsonify({
        "answer": answer,
        "confidence_score": 0.82,
        "recommended_pages": [
            {
                "title": p.get("title", "MIDC Page"),
                "url": p.get("source_url")
            }
            for p in pages[:3] if p.get("source_url")
        ],
        "external_links": [
            l.get("url") for l in external_links[:3] if l.get("url")
        ],
        "forms_detected": forms[:2],
        "conversation_state": {
            "intent": "midc",
            "should_follow_up": True,
            "follow_up_message":
                "Would you like help navigating a page or filling a related form?"
        }
    })

# =====================
# HEALTH CHECK
# =====================

@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot Backend Running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
