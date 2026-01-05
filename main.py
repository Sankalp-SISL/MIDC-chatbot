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

# Initialize Gen AI client (Vertex AI mode)
genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

app = Flask(__name__)

# =====================
# GCS HELPERS
# =====================

def load_section(section: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{section}/content.json")

    if not blob.exists():
        raise FileNotFoundError(f"No content.json for section: {section}")

    return json.loads(blob.download_as_text())


# =====================
# SECTION DETECTION
# =====================

def detect_sections(query: str):
    q = query.lower()
    sections = set()

    # ABOUT MIDC (critical fix)
    if any(k in q for k in [
        "about", "midc", "what is midc", "tell me about midc"
    ]):
        sections.add("about-midc")

    # Vision / Mission / Objectives
    if any(k in q for k in [
        "vision", "mission", "objective", "objectives"
    ]):
        sections.add("about-midc")

    # Maharashtra
    if "maharashtra" in q:
        sections.add("about-maharashtra")

    # Organisation / Departments
    if any(k in q for k in [
        "department", "organisation", "organization", "structure"
    ]):
        sections.add("departments-of-midc")

    # FAQ
    if any(k in q for k in ["faq", "frequently asked"]):
        sections.add("faq")

    # Investors
    if any(k in q for k in ["investor", "investment"]):
        sections.add("investors")

    # Customers
    if "customer" in q:
        sections.add("customers")

    # Focus sectors
    if any(k in q for k in ["sector", "industry", "focus"]):
        sections.add("focus-sectors")

    # Contact
    if any(k in q for k in ["contact", "address", "reach"]):
        sections.add("contact")

    # Notices
    if any(k in q for k in ["notice", "circular"]):
        sections.add("important-notice")

    # RTS / Acts
    if any(k in q for k in ["rts", "right to service", "act", "gazette"]):
        sections.update([
            "right-to-public-service-act",
            "rts-gazette",
            "list-of-services-under-rts-act"
        ])

    # Safe fallback
    if not sections:
        sections.add("about-midc")

    return list(sections)


# =====================
# CONTEXT BUILDER
# =====================

def build_context(sections):
    texts = []
    sources = []

    for section in sections:
        try:
            data = load_section(section)
            chunks = data.get("chunks", [])
            if chunks:
                texts.extend(chunks[:3])  # cap per section
                if data.get("source_url"):
                    sources.append(data["source_url"])
        except Exception as e:
            print(f"[WARN] Failed loading section '{section}': {e}")

    context = "\n\n".join(texts)
    return context, list(set(sources))


# =====================
# ROUTES
# =====================

@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        sections = detect_sections(question)
        context, sources = build_context(sections)

        # DEBUG (can be removed later)
        print("QUESTION:", question)
        print("SECTIONS:", sections)
        print("CONTEXT LENGTH:", len(context))
        print("SOURCES:", sources)

        prompt = f"""
You are an official information assistant for MIDC (Maharashtra Industrial Development Corporation).

Use the content below as the PRIMARY source.
If the content partially answers the question, summarize what is available.
Only say "The requested information is not available on MIDC's official website"
if the content is completely unrelated.

CONTENT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Be factual and concise
- Do not hallucinate
- Base answers strictly on the content
"""

        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )
            ]
        )

        answer_text = (
            response.text.strip()
            if response and response.text
            else "The requested information is not available on MIDC's official website."
        )

        return jsonify({
            "answer": answer_text,
            "sources": sources
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({
            "error": "Internal processing error",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
