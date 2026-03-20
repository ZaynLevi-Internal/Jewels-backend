"""
Aurielle Jewelry — AI Chatbot Backend
--------------------------------------
A lightweight Flask proxy that:
  • Keeps your Google API key server-side (never exposed to the browser)
  • Serves the static frontend (index.html)
  • Exposes a single POST /api/chat endpoint
  • Optionally saves leads to leads.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ── Configuration ──────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # allow requests from the same origin / dev servers

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Set this in your environment, not here in code!
LEADS_FILE = Path("leads.json")

# ── Knowledge base (extracted from index.html) ─────────────────────────────────

KNOWLEDGE_BASE = """
COMPANY: Aurielle Jewelry — Elegant Jewelry Boutique

TAGLINE: Timeless pieces, crafted with care.

DESCRIPTION: Artisanal necklaces, rings, and bespoke creations in gold, silver,
diamonds and platinum. Elegant designs made to last generations.

SHIPPING & WARRANTY:
- Free Shipping on orders above ₹10,000
- Lifetime Warranty on manufacturing faults

FEATURED COLLECTIONS / PRODUCTS:
1. Heritage Gold Necklace — Handcrafted 22k gold with filigree detail. ₹85,000
2. Solitaire Diamond Ring — Conflict-free brilliant cut diamond. ₹1,20,000
3. Classic Silver Bracelet — Sterling silver, polished finish. ₹9,500
4. Platinum Wedding Band — Comfort fit, elegant matte finish. ₹65,000
5. Pearl Drop Earrings — Freshwater pearls with minimal gold detail. ₹14,500
6. Bespoke Design Service — Work with our artisans to create your piece. From ₹5,000

SERVICES:
- Gold & Silver: Curated collections or custom pieces.
- Diamond Sourcing: Conflict-free diamonds with certification and appraisals.
- Repairs & Cleaning: Polishing, resizing, stone setting, maintenance.
- Bespoke Design: Design consultations and 3D previews.

ABOUT:
Founded by master artisans. Traditional techniques, modern design — ethical
sourcing, certified appraisals, in-house design studio.

TESTIMONIALS:
- Priya S.: "Beautiful craftsmanship and warm service. My engagement ring is
  everything I hoped for — and more."
- Amit R.: "Quick resizing and excellent attention to detail. Highly recommend
  their repair service."

CONTACT:
- Address: 123 Heritage Lane, Cityname
- Phone: +91 98765 43210
- Email: hello@aurielle.in

NAVIGATION: Collections → #collections | Services → #services |
            About → #about | Contact → #contact
"""

SYSTEM_PROMPT = f"""You are the AI assistant for Aurielle Jewelry, an elegant jewelry boutique.
Your name is "Aurielle Assistant".

Your role:
1. Answer questions accurately using ONLY the knowledge base below.
2. Help visitors navigate the site (mention section anchors like #collections).
3. Collect visitor leads when they show strong purchase or consultation intent —
   end your reply with [SHOW_LEAD_FORM] to trigger the lead-capture widget.
4. Handle FAQs about products, pricing, services, shipping, and repairs.

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

TONE: Warm, elegant, concise (2-4 sentences). Use ₹ for prices.
If asked something not in the knowledge base, direct the visitor to
+91 98765 43210 or hello@aurielle.in.

NEVER fabricate information not in the knowledge base.
"""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """API info endpoint."""
    return jsonify({"message": "Aurielle Jewelry API", "endpoints": ["/api/chat", "/api/leads"]})


@app.route("/api/chat", methods=["POST"])
def chat():
    
    if not GOOGLE_API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY is not set on the server."}), 500

    body = request.get_json(silent=True)
    if not body or "messages" not in body:
        return jsonify({"error": "Request body must include a 'messages' array."}), 400

    messages = body["messages"]

    # Basic validation
    for msg in messages:
        if msg.get("role") not in ("user", "assistant") or not isinstance(msg.get("content"), str):
            return jsonify({"error": "Invalid message format."}), 400

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            ),
            contents=contents
        )

        raw_reply = response.text
        show_lead_form = "[SHOW_LEAD_FORM]" in raw_reply
        clean_reply = raw_reply.replace("[SHOW_LEAD_FORM]", "").strip()

        return jsonify({"reply": clean_reply, "show_lead_form": show_lead_form})

    except Exception as exc:
        app.logger.error("Google API error: %s", exc)
        return jsonify({"error": "An unexpected error occurred."}), 500


@app.route("/api/leads", methods=["POST"])
def save_lead():
    """
    Saves visitor lead info to leads.json.
    Expects JSON: { "name": "...", "email": "...", "phone": "..." }
    In production, replace this with your CRM / database call.
    """
    data = request.get_json(silent=True) or {}
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()

    if not name or not email:
        return jsonify({"error": "name and email are required."}), 400

    lead = {
        "name": name,
        "email": email,
        "phone": phone,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Append to leads.json
    existing = []
    if LEADS_FILE.exists():
        try:
            existing = json.loads(LEADS_FILE.read_text())
        except json.JSONDecodeError:
            existing = []

    existing.append(lead)
    LEADS_FILE.write_text(json.dumps(existing, indent=2))

    app.logger.info("New lead: %s <%s>", name, email)
    return jsonify({"message": "Lead saved. Thank you!"}), 201


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("⚠️  WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("   Export it before running:  export GOOGLE_API_KEY=Az...")
    app.run(host="0.0.0.0", port=5000, debug=True)