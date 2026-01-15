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

ALLOWED_SECTIONS = [
    "about-midc",
    "about-maharashtra",
    "departments-of-midc",
    "faq",
    "investors",
    "customers",
    "country-desk",
    "focus-sectors",
    "contact",
    "important-notice",
    "right-to-public-service-act",
    "rts-gazette",
    "list-of-services-under-rts-act"
]

FOLLOW_UP_MAP = {
    "investor": "Would you like guidance on land availability, incentives, or the application process?",
    "land": "Do you want me to help you with land allotment steps or required documents?",
    "rts": "Would you like to know service timelines or documents required under the RTS Act?",
    "form": "Would you like me to help you fill the form step-by-step?",
    "general": "Can I help you with anything else on the MIDC website?"
}

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

app = Flask(__name__)

# =====================
# CORS (FOR BROWSER)
# =====================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
# GCS HELPERS
# =====================

def load_section(section: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{section}/content.json")

    if not blob.exists():
        return None

    return json.loads(blob.download_as_text())

def build_context(sections):
    texts = []
    sources = []

    for section in sections:
        data = load_section(section)
        if not data:
            continue

        chunks = data.get("chunks", [])
        if not chunks:
            continue

        if section in [
            "right-to-public-service-act",
            "rts-gazette",
            "list-of-services-under-rts-act"
        ]:
            texts.extend(chunks[:6])
        else:
            texts.extend(chunks[:3])

        if data.get("source_url"):
            sources.append(data["source_url"])

    return "\n\n".join(texts), list(set(sources))

# =====================
# SEMANTIC ROUTING
# =====================

def semantic_route_sections(question: str):
    routing_prompt = f"""
You are a routing assistant for an official government information system.

Allowed sections:
{", ".join(ALLOWED_SECTIONS)}

RULES:
- RTS / Act / Gazette questions MUST include:
  ["right-to-public-service-act", "rts-gazette", "list-of-services-under-rts-act"]
- Choose ONLY from allowed sections
- Return ONLY a JSON array
- If unsure, return ["about-midc"]

Question:
{question}
"""

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=routing_prompt)])],
            generation_config={"temperature": 0.0}
        )

        sections = json.loads(response.text.strip())
        valid = [s for s in sections if s in ALLOWED_SECTIONS]
        return valid if valid else ["about-midc"]

    except Exception:
        return ["about-midc"]

# =====================
# INTENT DETECTION
# =====================

def detect_intent(question: str, answer: str):
    prompt = f"""
Classify the user intent into ONE of the following:
- investor
- land
- rts
- form
- general

Return ONLY valid JSON:
{{ "intent": "<intent>" }}

Question:
{question}

Answer:
{answer}
"""

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            generation_config={"temperature": 0.0}
        )
        data = json.loads(response.text.strip())
        return data.get("intent", "general")
    except Exception:
        return "general"

# =====================
# ROUTES
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
    language_instruction = (
        "Respond in Marathi." if language == "mr"
        else "Respond in English."
    )

    sections = semantic_route_sections(question)
    context, sources = build_context(sections)

    prompt = f"""
You are an official information assistant for MIDC (Maharashtra Industrial Development Corporation).

CONTENT:
{context}

QUESTION:
{question}

RULES:
- Answer strictly from the provided content
- Do not hallucinate
- If information is missing, say it clearly
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer_text = (
        response.text.strip()
        if response and response.text
        else "The requested information is not available on MIDC's official website."
    )

    intent = detect_intent(question, answer_text)
    follow_up = FOLLOW_UP_MAP.get(intent)

    return jsonify({
        "answer": answer_text,
        "sources": sources,
        "conversation_state": {
            "intent": intent,
            "should_follow_up": True,
            "follow_up_message": follow_up
        }
    })

@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot backend is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
