import os
import json
import time
import re
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
# 🔒 MIDC ENTITY & QUERY CLASSIFIERS
# =====================

MIDC_ENTITY_KEYWORDS = [
    "ceo", "managing director", "md",
    "chairman", "contact", "email",
    "phone", "officer", "official",
    "helpline", "address", "head office"
]

INTERNET_EXPLICIT_KEYWORDS = [
    "search internet",
    "search web",
    "use internet",
    "google this",
    "outside midc"
]

def is_midc_entity_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in MIDC_ENTITY_KEYWORDS)

def is_explicit_internet_query(question: str, mode: str | None) -> bool:
    if mode == "internet":
        return True
    q = question.lower()
    return any(k in q for k in INTERNET_EXPLICIT_KEYWORDS)

# =====================
# 🧹 LLM OUTPUT CLEANER (NEW – CRITICAL)
# =====================

def clean_llm_html(text: str) -> str:
    if not text:
        return text

    # Remove markdown fences if model adds them
    text = re.sub(r"^```html", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    return text.strip()

# =====================
# LOAD GCS CONTENT
# =====================

def load_all_content():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    pages, pdfs, forms, external_links = [], [], [], []

    for blob in bucket.list_blobs():
        if not blob.name.endswith(".json"):
            continue

        try:
            data = json.loads(blob.download_as_text())
            content_type = data.get("content_type", "html")

            if content_type == "pdf":
                pdfs.append(data)
            else:
                pages.append(data)

            external_links.extend(data.get("related_links", []))
            forms.extend(data.get("forms", []))

        except Exception:
            continue

    return pages, pdfs, forms, external_links

# =====================
# CONTEXT BUILDER (BOOSTED, SAFE)
# =====================

CONTACT_TERMS = [
    "contact", "email", "phone",
    "address", "ceo", "md",
    "helpline", "office"
]

def build_context(pages, pdfs, question: str):
    q = question.lower()
    chunks = []

    # 🔒 HARD PRIORITY: CONTACT / CEO
    if any(k in q for k in CONTACT_TERMS):
        for p in pages:
            if "contact" in (p.get("section", "") or "").lower():
                chunks.extend(p.get("chunks", [])[:6])

    # 🔹 KEYWORD-SCORED PAGES
    scored = []
    for p in pages:
        title = (p.get("section") or "").replace("-", " ").lower()
        score = sum(1 for w in q.split() if w in title)
        if score > 0:
            scored.append((score, p))

    scored.sort(reverse=True, key=lambda x: x[0])

    for _, p in scored[:5]:
        chunks.extend(p.get("chunks", [])[:3])

    # 🔹 FALLBACK
    if len(chunks) < 6:
        for p in pages[:5]:
            chunks.extend(p.get("chunks", [])[:2])

    # 🔹 PDFs LAST
    for p in pdfs[:3]:
        chunks.extend(p.get("chunks", [])[:2])

    return "\n\n".join(chunks)

# =====================
# 🔗 RECOMMENDED PAGE MATCHER (NEW)
# =====================

def recommend_pages(question: str, pages):
    q = question.lower()
    scored = []

    for p in pages:
        section = (p.get("section") or "").replace("-", " ").lower()
        score = sum(1 for w in q.split() if w in section)
        if score > 0 and p.get("source_url"):
            scored.append((score, p))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [
        {
            "title": p.get("section", "MIDC Page").replace("-", " ").title(),
            "url": p.get("source_url")
        }
        for _, p in scored[:5]
    ]

# =====================
# INTERNET ANSWER (EXPLICIT ONLY – SAFE)
# =====================

def internet_answer(question, language_instruction):
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(
            role="user",
            parts=[types.Part(
                text=f"""
Answer using general public knowledge.

IMPORTANT:
- This is NOT official MIDC information
- Clearly state this disclaimer
- Do NOT hallucinate
- Output VALID HTML ONLY
- Use <strong>, <ul>, <li>, <a>

{language_instruction}

QUESTION:
{question}
"""
            )]
        )]
    )

    return clean_llm_html(response.text) if response and response.text else None

# =====================
# CHAT ENDPOINT
# =====================

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    mode = body.get("mode")

    if not question:
        return jsonify({"error": "Question required"}), 400

    language = detect_language(question)
    language_instruction = (
        "Respond in Marathi." if language == "mr"
        else "Respond in English."
    )

    # 🔒 MIDC ENTITY ALWAYS OVERRIDES INTERNET
    force_midc = is_midc_entity_query(question)

    # 🌐 INTERNET MODE (ONLY IF EXPLICIT)
    if not force_midc and is_explicit_internet_query(question, mode):
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

    # 🏛 MIDC MODE (PRIMARY)
    pages, pdfs, forms, external_links = load_all_content()
    context = build_context(pages, pdfs, question)

    prompt = f"""
You are an official AI assistant for
Maharashtra Industrial Development Corporation (MIDC).

CONTENT:
{context}

QUESTION:
{question}

MANDATORY RULES:
- OUTPUT VALID HTML ONLY
- NO markdown
- NO '*' or '**'
- Use <strong>, <ul>, <li>, <a>
- Extract factual information directly
- If contact-related, show phone/email/address clearly
- Link the exact MIDC page
- Do NOT hallucinate
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer = clean_llm_html(
        response.text.strip()
        if response and response.text
        else "<strong>Information not available on the MIDC website.</strong>"
    )

    return jsonify({
        "answer": answer,
        "confidence_score": 0.82,
        "recommended_pages": recommend_pages(question, pages),
        "external_links": [
            {
                "title": l.get("title", "External Link"),
                "url": l.get("url")
            }
            for l in external_links
            if l.get("url")
        ][:5],
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
