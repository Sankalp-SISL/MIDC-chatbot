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

# Allowed content sections (authoritative list)
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

# Initialize Gemini (Vertex AI)
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
        raise FileNotFoundError(f"Missing content.json for section: {section}")

    return json.loads(blob.download_as_text())


def build_context(sections):
    texts = []
    sources = []

    for section in sections:
        try:
            data = load_section(section)
            chunks = data.get("chunks", [])

            if chunks:
                texts.extend(chunks[:3])  # cap chunks per section
                if data.get("source_url"):
                    sources.append(data["source_url"])

        except Exception:
            continue

    return "\n\n".join(texts), list(set(sources))


# =====================
# SEMANTIC ROUTING (GEMINI-BASED)
# =====================

def semantic_route_sections(question: str):
    """
    Uses Gemini to semantically determine relevant sections.
    Returns a safe subset of ALLOWED_SECTIONS.
    """

    routing_prompt = f"""
You are a routing assistant for an official government information system.

From the list of allowed sections below, choose the MOST relevant sections
for answering the user question.

Allowed sections:
{", ".join(ALLOWED_SECTIONS)}

Rules:
- Choose ONLY from the allowed sections
- Return ONLY a valid JSON array (no text)
- If unsure, return ["about-midc"]

User question:
{question}
"""

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=routing_prompt)]
                )
            ],
            generation_config={
                "temperature": 0.0
            }
        )

        sections = json.loads(response.text.strip())

        if isinstance(sections, list):
            valid_sections = [s for s in sections if s in ALLOWED_SECTIONS]
            return valid_sections if valid_sections else ["about-midc"]

    except Exception:
        pass

    return ["about-midc"]


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

        # Semantic routing
        sections = semantic_route_sections(question)

        # Load grounded context
        context, sources = build_context(sections)

        prompt = f"""
You are an official information assistant for MIDC (Maharashtra Industrial Development Corporation).

Use the content below as the PRIMARY source.
If the content partially answers the question, summarize what is available.
Only respond with:
"The requested information is not available on MIDC's official website."
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
        return jsonify({
            "error": "Internal processing error",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return "MIDC Chatbot is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
