import os
import json
import time
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

FOLLOW_UP_IDLE_SECONDS = 30  # auto follow-up window

# RTS documents are mandatory context
RTS_SECTIONS = [
    "right-to-public-service-act",
    "rts-gazette",
    "list-of-services-under-rts-act"
]

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# =========================================================
# CORS (Browser Safe)
# =========================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
# INTENT & RTS DETECTION
# =========================================================

def is_rts_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "rts",
        "right to service",
        "right to public service",
        "gazette",
        "public service act"
    ])

# =========================================================
# GCS LOADERS
# =========================================================

def load_json_blob(blob_name: str):
    blob = bucket.blob(blob_name)
    if blob.exists():
        return json.loads(blob.download_as_text())
    return None

def load_all_documents(limit=50):
    """Loads crawler-generated page & pdf JSONs"""
    docs = []
    for blob in bucket.list_blobs():
        if blob.name.endswith(".json"):
            try:
                docs.append(json.loads(blob.download_as_text()))
            except Exception:
                continue
        if len(docs) >= limit:
            break
    return docs

def load_rts_documents():
    docs = []
    for sec in RTS_SECTIONS:
        data = load_json_blob(f"{sec}/content.json")
        if data:
            docs.append(data)
    return docs

# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(docs):
    text_chunks = []
    pages = []
    links = []
    forms = []

    for d in docs:
        chunks = d.get("chunks", [])
        text_chunks.extend(chunks[:5])

        if d.get("source_url"):
            pages.append({
                "title": d.get("section", "MIDC Page"),
                "url": d["source_url"]
            })

        links.extend(d.get("external_links", []))
        forms.extend(d.get("forms", []))

    return (
        "\n\n".join(text_chunks),
        pages,
        links,
        forms
    )

# =========================================================
# SEMANTIC FILTERING (NON-RTS)
# =========================================================

def semantic_filter(question: str, docs):
    prompt = f"""
From the following documents, select the ones relevant to the question.

Return ONLY a JSON list of indices.

Question:
{question}
"""

    doc_snippets = [
        f"[{i}] {d.get('section','')} {(' '.join(d.get('chunks',[])[:1]))}"
        for i, d in enumerate(docs)
    ]

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=prompt + "\n\n" + "\n".join(doc_snippets))]
            )],
            generation_config={"temperature": 0.0}
        )

        idxs = json.loads(response.text.strip())
        return [docs[i] for i in idxs if i < len(docs)]

    except Exception:
        return docs[:3]

# =========================================================
# CONFIDENCE SCORE
# =========================================================

def compute_confidence(answer: str, context: str) -> float:
    if not answer or "not available" in answer.lower():
        return 0.1
    overlap = len(set(answer.split()) & set(context.split()))
    return round(min(1.0, 0.3 + overlap / 200), 2)

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    last_ts = body.get("last_interaction_ts")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    language = detect_language(question)
    language_instruction = (
        "Respond in Marathi." if language == "mr"
        else "Respond in English."
    )

    # ---------- Document selection ----------
    if is_rts_query(question):
        docs = load_rts_documents()
        intent = "RTS"
    else:
        all_docs = load_all_documents()
        docs = semantic_filter(question, all_docs)
        intent = "general"

    context, pages, links, forms = build_context(docs)

    # ---------- Prompt ----------
    prompt = f"""
You are an official AI assistant for the MIDC website.

Use ONLY the provided content.
If pages or links exist, recommend them explicitly.
If forms exist, guide the user step-by-step.

CONTENT:
{context}

QUESTION:
{question}

RULES:
- No hallucination
- Be helpful and actionable
- {language_instruction}
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    answer = (
        response.text.strip()
        if response and response.text
        else "The requested information is not available on MIDC's official website."
    )

    confidence = compute_confidence(answer, context)

    # ---------- Auto follow-up ----------
    follow_up = None
    should_follow = False
    now = int(time.time())

    if confidence > 0.3:
        should_follow = True
        follow_up = (
            "Would you like me to guide you to the exact page or help with a form?"
            if intent != "RTS"
            else "Would you like to know applicable RTS services and timelines?"
        )

    return jsonify({
        "answer": answer,
        "confidence_score": confidence,
        "recommended_pages": pages[:5],
        "external_links": links[:5],
        "forms_detected": forms,
        "conversation_state": {
            "intent": intent,
            "should_follow_up": should_follow,
            "follow_up_message": follow_up,
            "last_interaction_ts": now
        }
    })

# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def health():
    return "MIDC Agentic Chatbot is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
