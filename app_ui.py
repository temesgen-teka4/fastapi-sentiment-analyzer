# ==========================================
# STREAMLIT TEXT INTELLIGENCE DASHBOARD (v4.1 - Cloud Optimized)
# ==========================================
import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="AI Text Intelligence Dashboard",
    page_icon="🧠",
    layout="centered"
)

# Fetch API Key from environment or Streamlit Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

st.title("🧠 AI Text Intelligence Platform")
st.caption("Powered by Gemini 2.5 Flash Reasoning Engine • Human-Level Intent & Sarcasm Decoding")

with st.sidebar:
    st.header("⚙️ Configuration")
    if client:
        st.success("🟢 Gemini API Connected")
    else:
        st.warning("🟡 API Key Missing")
        user_key = st.text_input("Enter Gemini API Key:", type="password")
        if user_key:
            client = genai.Client(api_key=user_key)
            st.rerun()

st.markdown("---")

default_text = "If the customer support team were actually responsive, this would easily be a 5-star product."
user_text = st.text_area("Paste your input text below:", value=default_text, height=130)

if st.button("🚀 Analyze with Deep AI Reasoning", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    elif not client:
        st.error("Please provide a valid Gemini API Key in the sidebar or Streamlit Secrets.")
    else:
        clean_input = user_text.strip().strip('"').strip("'")
        
        with st.spinner("Executing Deep Reasoning AI Analysis..."):
            try:
                prompt = f"""
Analyze the text below with deep human-level comprehension. Account for sarcasm, double negatives, counterfactual conditionals, clause structures, and implicit subtext.

Text: "{clean_input}"

Return ONLY a raw JSON object with this exact structure:
{{
    "overall_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence_score": float between 0.0 and 1.0,
    "is_sarcastic_or_conditional": boolean,
    "detected_nuance": "Brief label (e.g. Counterfactual Criticism, Sarcasm, Direct Praise)",
    "human_reasoning": "A 1-2 sentence explanation of what the speaker ACTUALLY means in reality.",
    "clause_breakdown": [
        {{
            "chunk_id": 1,
            "text": "segment or clause text",
            "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "confidence": float between 0.0 and 1.0
        }}
    ]
}}
"""
                res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                llm_data = json.loads(res.text)
                
                st.session_state["analysis_data"] = {
                    "overall_sentiment": llm_data["overall_sentiment"].upper(),
                    "confidence_score": llm_data["confidence_score"],
                    "is_sarcastic": llm_data["is_sarcastic_or_conditional"],
                    "nuance": llm_data["detected_nuance"],
                    "human_reasoning": llm_data["human_reasoning"],
                    "clause_breakdown": llm_data.get("clause_breakdown", [])
                }
            except Exception as e:
                st.error(f"API Error during analysis: {e}")

# --- DISPLAY RESULTS ---
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emojis = {"POSITIVE": "POSITIVE 😃", "NEGATIVE": "NEGATIVE 🙁", "NEUTRAL": "NEUTRAL 😐"}
        st.metric("Overall Sentiment", emojis.get(data["overall_sentiment"], data["overall_sentiment"]))
        
    with col2:
        st.metric("Confidence Score", f"{data['confidence_score'] * 100:.1f}%")
        
    with col3:
        st.metric("Detected Tone Nuance", data["nuance"])

    st.info(f"💡 **AI Human Reasoning Insight**:\n\n\"{data['human_reasoning']}\"")

    if data["is_sarcastic"]:
        st.warning("⚠️ **Nuance Alert**: Counterfactual or sarcastic subtext detected.")

    st.markdown("---")
    st.markdown("### 🔍 Sentence Clause Breakdown")
    for chunk in data["clause_breakdown"]:
        lab = chunk['sentiment']
        badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
        with st.expander(f"Clause #{chunk['chunk_id']}: {badge} ({chunk['confidence']*100:.1f}%)", expanded=True):
            st.write(f"**Text:** *\"{chunk['text']}\"*")
            st.progress(chunk['confidence'])