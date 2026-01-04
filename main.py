import os
import json
from flask import Flask, request, jsonify
from google.cloud import storage

from google import genai
from google.genai import types

# --------------------
# CONFIG
# --------------------
BUCKET_NAME = "midc-chatbot-content"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

# Initialize GenAI client for Vertex AI
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

app = Flask(__name__)

# --------------------
# HELPERS
# --------------------
def load_section(section: str):
    client_gcs = storage.Client()
    bucket = client_gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{section}/content.json")
    return json.loads(blob.download_as_text())

def detect_sections(query: str):
    q = query.lower()
    sections = set()

    if any(k in q for k in ["rts", "right to service", "act", "gazette"]):
        sections.update([
            "right-to-public-service-act",
            "rts-gazette",
            "list-of-services-under-rts-act"
        ])

    if any(k in q for k in ["vision", "mission"]):
        sections.add("about-midc")
    if "maharashtra" in q:
        sections.add("about-maharashtra")
    if "faq" in q:
        sections.add("faq")
    if "investor" in q:
        sections.add("investors")

    if not sections:
        sections.update(["about-midc", "faq"])

    return list(sections)

def build_context(sections):
    texts = []
    sources = []

    for s in sections:
        try:
            data = load_section(s)
            texts.extend(data.get("chunks", [])[:3])
            sources.append(data.get("source_url"))
        except Exception as e:
            print(f"[WARN] Failed loading {s}: {e}")

    return "\n\n".join(texts), list(set(sources))

# --------------------
# ROUTES
# --------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        sections = detect_sections(question)
        context, sources = build_context(sections)

        prompt = f"""
You are an official information assistant for MIDC (Maharashtra Industrial Development Corporation).

Answer ONLY using the content provided below.
If the answer is not present, respond exactly with:
"The requested information is not available on MIDC's official website."

CONTENT:
{context}

QUESTION:
{question}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )
            ]
        )

        text = response.text if response and response.text else (
            "The requested information is not available on MIDC's official website."
        )

        return jsonify({
            "answer": text,
            "sources": sources
        })

    except Exception as e:
        print("[FATAL ERROR]", str(e))
        return jsonify({
            "error": "Internal processing error",
            "details": str(e)
        }), 500

@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
