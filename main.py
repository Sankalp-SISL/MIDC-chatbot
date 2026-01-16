import os
import json
import time
from typing import List, Dict

from flask import Flask, request, jsonify
from google.cloud import storage
from google import genai
from google.genai import types

# =========================================================
# CONFIGURATION
# =========================================================

BUCKET_NAME = "midc-general-chatbot-bucket-web-data"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.6

INTERNET_TRIGGER_KEYWORDS = [
    "search internet",
    "search web",
    "outside midc",
    "general knowledge",
    "latest",
    "news",
    "google"
]

# =========================================================
# CLIENTS
# =========================================================

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

gcs_client = storage.Client()

# =========================================================
# APP
# =========================================================

app = Flask(__name__)

# =========================================================
# CORS
# =========================================================

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text: str) -> str:
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x097F:
            return "mr"
    return "en"

# =========================================================
# INTERNET MODE DETECTION
# =========================================================

def is_internet_query(question: str, mode: str | None) -> bool:
    if mode == "internet":
        return True
    q = question.lower()
    return any(k in q for k in INTERNET_TRIGGER_KEYWORDS)

# =========================================================
# LOAD CRAWLED CONTENT FROM GCS
# =========================================================

def load_all_content():
    bucket = gcs_client.bucket(BUCKET_NAME)

    pages, pdfs, forms, external_links = [], [], [], []

    for blob in bucket.list_blobs():
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

# =========================================================
# CONTEXT BUILDER (PAGE-LEVEL GRAPH)
# =========================================================

def build_context(pages: List[Dict], pdfs: List[Dict]) -> str:
    chunks = []

    for p in pages[:6]:
        chunks.extend(p.get("chunks", [])[:3])

    for p in pdfs[:4]:
        chunks.extend(p.get("chunks", [])[:2])

    return "\n\n".join(chunks)

# =========================================================
# FORM MATCHER
# =========================================================

def detect_relevant_forms(question: str, forms: List[Dict]):
    matches = []
    q = question.lower()

    for f in forms:
        keywords = f.get("metadata", {}).get("keywords", [])
        if any(k.lower() in q for k in keywords):
            matches.append(f)

    return matches

# =========================================================
# PAGE / LINK MATCHER
# =========================================================

def recommend_pages(question: str, pages: List[Dict]):
    q = question.lower()
    results = []

    for p in pages:
        title = p.get("title", "")
        if title and any(w in q for w in title.lower().split()):
            results.append({
                "title": title,
                "url": p.get("source_url")
            })

    return results[:5]

# =========================================================
# INTERNET ANSWER (SAFE)
# =========================================================

def internet_answer(question: str, language_instruction: str):
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(
                    text=f"""
Use Google Search to answer this question.
This is NOT official MIDC information.
Do NOT hallucinate.
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

# =========================================================
# MAIN CHAT ROUTE
# =========================================================

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    mode = body.get("mode")
    last_ts = body.get("last_interaction_ts")

    if not question:
        return jsonify({"error": "Question required"}), 400

    language = detect_language(question)
    language_instruction = "Respond in Marathi." if language == "mr" else "Respond in English."

    # =====================================================
    # INTERNET MODE
    # =====================================================

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

    # =====================================================
    # MIDC MODE
    # =====================================================

    pages, pdfs, forms, external_links = load_all_content()
    context = build_context(pages, pdfs)

    matched_forms = detect_relevant_forms(question, forms)
    recommended = recommend_pages(question, pages)

    prompt = f"""
You are an official AI assistant for MIDC.

CONTENT:
{context}

QUESTION:
{question}

RULES:
- Answer strictly from MIDC content
- If a FORM exists, guide user step-by-step
- If an EXTERNAL LINK exists, expose it clearly
- Recommend exact pages when possible
- Do NOT hallucinate
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer = response.text.strip() if response and response.text else None

    confidence = (
        CONFIDENCE_HIGH if answer else 0.0
    )

    follow_up = None
    should_follow_up = False

    if matched_forms:
        follow_up = "I found a form related to this. Would you like me to help you fill it step by step?"
        should_follow_up = True
    elif not answer:
        follow_up = "I could not find this on MIDC. Should I search the internet for you?"
        should_follow_up = True

    return jsonify({
        "answer": answer or "The requested information is not available on MIDC's official website.",
        "confidence_score": round(confidence, 2),
        "sources": list({
            p.get("source_url")
            for p in pages if p.get("source_url")
        }),
        "external_links": [
            {
                "title": l.get("title"),
                "url": l.get("url")
            } for l in external_links[:5]
        ],
        "forms_detected": matched_forms,
        "recommended_pages": recommended,
        "conversation_state": {
            "intent": "midc",
            "should_follow_up": should_follow_up,
            "follow_up_message": follow_up,
            "last_interaction_ts": int(time.time())
        }
    })

# =========================================================
# HEALTH
# =========================================================

@app.route("/", methods=["GET"])
def health():
    return "MIDC AI Assistant Backend Running", 200

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
