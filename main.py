import os
import json
import time
from flask import Flask, request, jsonify
from google.cloud import storage
from google import genai
from google.genai import types

BUCKET = "midc-general-chatbot-bucket-web-data"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-001"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
app = Flask(__name__)

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return resp


def detect_lang(text):
    return "mr" if any(0x0900 <= ord(c) <= 0x097F for c in text) else "en"


def load_all():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    docs = []

    for blob in bucket.list_blobs():
        docs.append(json.loads(blob.download_as_text()))
    return docs


DOCS = load_all()


def semantic_match(question):
    prompt = f"""
Select relevant documents to answer:
{question}

Return JSON array of indices.
"""
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        generation_config={"temperature": 0}
    )
    try:
        return json.loads(r.text)
    except:
        return []


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    q = request.json.get("question", "").strip()
    if not q:
        return jsonify({"error": "Empty question"}), 400

    lang = detect_lang(q)
    doc_idxs = semantic_match(q)

    context = []
    pages = []
    forms = []
    links = []

    for i in doc_idxs[:6]:
        try:
            d = DOCS[i]
            context.extend(d.get("chunks", [])[:2])
            if d.get("source_url"):
                pages.append({
                    "title": d["section"].replace("-", " ").title(),
                    "url": d["source_url"]
                })
            forms.extend(d.get("forms", []))
            links.extend(d.get("related_links", []))
        except:
            pass

    prompt = f"""
You are an official MIDC website assistant.

Answer strictly from context.
If action exists, guide user.

Context:
{chr(10).join(context)}

Question:
{q}

Language: {"Marathi" if lang=="mr" else "English"}
"""

    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )

    confidence = min(1.0, len(context) / 10)

    follow_up = None
    if forms:
        follow_up = "I can help you fill the required form step-by-step. Would you like to proceed?"
    elif pages:
        follow_up = "Would you like me to open the relevant page or guide you further?"

    return jsonify({
        "answer": resp.text.strip(),
        "confidence_score": round(confidence, 2),
        "recommended_pages": pages,
        "forms_detected": forms,
        "external_links": links,
        "conversation_state": {
            "should_follow_up": True if follow_up else False,
            "follow_up_message": follow_up
        }
    })


@app.route("/", methods=["GET"])
def health():
    return "MIDC Agentic Assistant running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
