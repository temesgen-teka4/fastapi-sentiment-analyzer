# ==========================================
# STREAMLIT SENTIMENT ANALYZER DASHBOARD (v4.0)
# ==========================================
import os
import json
import re
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="AI Text Intelligence Dashboard v4.0",
    page_icon="🧠",
    layout="centered"
)

# Fetch API Key from environment or Streamlit Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def load_fallback_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

with st.spinner("Initializing AI Core..."):
    fallback_pipeline = load_fallback_model()

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

st.title("🧠 AI Text Intelligence Platform")
st.caption("Powered by Gemini 2.5 Flash Reasoning Engine • Human-Level Intent & Sarcasm Decoding")

# API Key Notice / Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    if client:
        st.success("🟢 Deep Reasoning Engine Active (Gemini API Connected)")
    else:
        st.warning("🟡 RoBERTa Baseline Active (Set GEMINI_API_KEY for Deep Reasoning)")
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
    else:
        clean_input = user_text.strip().strip('"').strip("'")
        
        # 1. Structural Clause Breakdown
        raw_chunks = re.split(r'(?<=[.!?])\s+|,\s+(?=but|however|although|yet)', clean_input, flags=re.IGNORECASE)
        sentences = [c.strip() for c in raw_chunks if c.strip()]
        
        clause_breakdown = []
        for i, s in enumerate(sentences):
            p = fallback_pipeline(s[:512])[0]
            clause_breakdown.append({
                "chunk_id": i + 1,
                "text": s,
                "sentiment": p["label"].upper(),
                "confidence": p["score"]
            })

        # 2. Deep Reasoning Engine Execution
        if client:
            with st.spinner("Executing Deep Reasoning LLM Analysis..."):
                try:
                    prompt = f"""
Analyze the text below with deep human-level comprehension. Account for sarcasm, double negatives, counterfactual conditionals, and implicit subtext.

Text: "{clean_input}"

Return ONLY a raw JSON object with this exact structure:
{{
    "overall_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence_score": float between 0.0 and 1.0,
    "is_sarcastic_or_conditional": boolean,
    "detected_nuance": "Brief label (e.g. Counterfactual Criticism, Sarcasm, Direct Praise)",
    "human_reasoning": "A 1-2 sentence explanation of what the speaker ACTUALLY means in reality."
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
                    
                    analysis_data = {
                        "engine": "Gemini 2.5 Flash (Deep Reasoning)",
                        "overall_sentiment": llm_data["overall_sentiment"].upper(),
                        "confidence_score": llm_data["confidence_score"],
                        "is_sarcastic": llm_data["is_sarcastic_or_conditional"],
                        "nuance": llm_data["detected_nuance"],
                        "human_reasoning": llm_data["human_reasoning"],
                        "clause_breakdown": clause_breakdown
                    }
                except Exception as e:
                    st.error(f"Reasoning API Error: {e}")
                    analysis_data = None
        else:
            # Fallback RoBERTa calculation
            total_score = sum(c["confidence"] if c["sentiment"] == "POSITIVE" else (-c["confidence"] if c["sentiment"] == "NEGATIVE" else 0.0) for c in clause_breakdown)
            avg = total_score / len(clause_breakdown) if clause_breakdown else 0.0
            overall = "POSITIVE" if avg > 0.15 else ("NEGATIVE" if avg < -0.15 else "NEUTRAL")
            
            analysis_data = {
                "engine": "RoBERTa Baseline Pipeline",
                "overall_sentiment": overall,
                "confidence_score": abs(avg),
                "is_sarcastic": False,
                "nuance": "Statistical Pattern Match",
                "human_reasoning": "Baseline statistical evaluation executed without LLM contextual reasoning.",
                "clause_breakdown": clause_breakdown
            }

        if analysis_data:
            st.session_state["analysis_data"] = analysis_data

# --- DISPLAY RESULTS ---
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    
    st.markdown(f"### 📊 Analysis Summary (`{data['engine']}`)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emojis = {"POSITIVE": "POSITIVE 😃", "NEGATIVE": "NEGATIVE 🙁", "NEUTRAL": "NEUTRAL 😐"}
        st.metric("Overall Sentiment", emojis.get(data["overall_sentiment"], data["overall_sentiment"]))
        
    with col2:
        st.metric("Confidence Score", f"{data['confidence_score'] * 100:.1f}%")
        
    with col3:
        st.metric("Detected Tone Nuance", data["nuance"])

    # LLM Human Reasoning Box
    st.info(f"💡 **AI Human Reasoning Insight**:\n\n\"{data['human_reasoning']}\"")

    if data["is_sarcastic"]:
        st.warning("⚠️ **Nuance Alert**: Counterfactual or sarcastic subtext detected. The true sentiment reflects present reality rather than literal surface keywords.")

    st.markdown("---")
    st.markdown("### 🔍 Structural Clause Breakdown")
    for chunk in data["clause_breakdown"]:
        lab = chunk['sentiment']
        badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
        with st.expander(f"Clause #{chunk['chunk_id']}: {badge} ({chunk['confidence']*100:.1f}%)", expanded=True):
            st.write(f"**Text:** *\"{chunk['text']}\"*")
            st.progress(chunk['confidence'])