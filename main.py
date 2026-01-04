import os
import json
from flask import Flask, request, jsonify
from google.cloud import storage
import google.generativeai as genai

# --------------------
# CONFIG
# --------------------
BUCKET_NAME = "midc-chatbot-content"
MODEL_NAME = "gemini-1.5-flash"

# Gemini via service account (no API key needed)
genai.configure()  # uses default credentials in Cloud Run

app = Flask(__name__)

# --------------------
# HELPERS
# --------------------
def load_section(section: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{section}/content.json")
    data = json.loads(blob.download_as_text())
    return data

def detect_sections(query: str):
    q = query.lower()
    sections = set()

    # RTS / PDFs
    if any(k in q for k in ["rts", "right to service", "act", "gazette"]):
        sections.update([
            "right-to-public-service-act",
            "rts-gazette",
            "list-of-services-under-rts-act"
        ])

    # Common pages
    if any(k in q for k in ["vision", "mission"]):
        sections.add("about-midc")
    if "maharashtra" in q:
        sections.add("about-maharashtra")
    if any(k in q for k in ["department", "organisation", "organization", "structure"]):
        sections.add("departments-of-midc")
    if "faq" in q:
        sections.add("faq")
    if "investor" in q:
        sections.add("investors")
    if "customer" in q:
        sections.add("customers")
    if "sector" in q:
        sections.add("focus-sectors")
    if "contact" in q:
        sections.add("contact")
    if "notice" in q:
        sections.add("important-notice")

    # Safe fallback
    if not sections:
        sections.update(["about-midc", "faq"])

    return list(sections)

def build_context(sections):
    texts = []
    sources = []

    for s in sections:
        try:
            data = load_section(s)
            texts.extend(data.get("chunks", [])[:3])  # cap per section
            sources.append(data.get("source_url"))
        except Exception:
            continue

    context = "\n\n".join(texts)
    return context, list(set(sources))

# --------------------
# ROUTES
# --------------------
@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    sections = detect_sections(question)
    context, sources = build_context(sections)

    prompt = f"""
You are an official information assistant for MIDC.
Answer ONLY using the provided content.
If the answer is not present, say:
"The requested information is not available on MIDC's official website."

CONTENT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Be concise and factual
- Do not infer or add external knowledge
- Cite sources at the end
"""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    return jsonify({
        "answer": response.text,
        "sources": sources
    })

@app.route("/")
def health():
    return "MIDC Chatbot is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))