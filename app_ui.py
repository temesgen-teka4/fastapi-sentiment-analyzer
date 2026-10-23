# ==========================================
# STREAMLIT TEXT INTELLIGENCE DASHBOARD (v4.4)
# ==========================================
import os
import json
import streamlit as st
import google.generativeai as genai

st.set_page_config(
    page_title="AI Text Intelligence Dashboard",
    page_icon="🧠",
    layout="centered"
)

# Safely fetch API key
api_key_env = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key_env = st.secrets["GEMINI_API_KEY"]
    elif os.getenv("GEMINI_API_KEY"):
        api_key_env = os.getenv("GEMINI_API_KEY")
except Exception:
    pass

st.title("🧠 AI Text Intelligence Platform")
st.caption("Powered by Gemini 3.8 Flash • Human-Level Intent & Sarcasm Decoding")

with st.sidebar:
    st.header("⚙️ Configuration")
    user_input_key = st.text_input("Gemini API Key:", value=api_key_env, type="password")
    if user_input_key:
        st.success("🟢 API Key Ready")
    else:
        st.warning("🟡 Please paste your Gemini API key.")

st.markdown("---")

default_text = "If the customer support team were actually responsive, this would easily be a 5-star product."
user_text = st.text_area("Paste your input text below:", value=default_text, height=130)

if st.button("🚀 Analyze with Deep AI Reasoning", use_container_width=True):
    active_key = user_input_key.strip()
    
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    elif not active_key:
        st.error("❌ Error: No Gemini API Key found. Please enter it in the sidebar.")
    else:
        clean_input = user_text.strip().strip('"').strip("'")
        
        with st.spinner("Analyzing with Gemini 3.8 Flash..."):
            try:
                genai.configure(api_key=active_key)
                # Use current stable Gemini 3.8 Flash model identifier
                model = genai.GenerativeModel('gemini-3.8-flash')
                
                prompt = f"""
Analyze the text below with deep human-level comprehension. Account for sarcasm, double negatives, counterfactual conditionals, clause structures, and implicit subtext.

Text: "{clean_input}"

Return ONLY a raw JSON object (no markdown formatting blocks like ```json, just the pure JSON text) with this exact schema:
{{
    "overall_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence_score": 0.95,
    "is_sarcastic_or_conditional": true,
    "detected_nuance": "Counterfactual Criticism",
    "human_reasoning": "A 1-2 sentence explanation of what the speaker ACTUALLY means in reality.",
    "clause_breakdown": [
        {{
            "chunk_id": 1,
            "text": "{clean_input}",
            "sentiment": "NEGATIVE",
            "confidence": 0.95
        }}
    ]
}}
"""
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
                )
                
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                
                llm_data = json.loads(raw_text.strip())
                
                st.session_state["analysis_data"] = {
                    "overall_sentiment": llm_data.get("overall_sentiment", "NEUTRAL").upper(),
                    "confidence_score": float(llm_data.get("confidence_score", 0.0)),
                    "is_sarcastic": bool(llm_data.get("is_sarcastic_or_conditional", False)),
                    "nuance": llm_data.get("detected_nuance", "Standard Analysis"),
                    "human_reasoning": llm_data.get("human_reasoning", "No reasoning provided."),
                    "clause_breakdown": llm_data.get("clause_breakdown", [])
                }
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Execution Error: {str(e)}")

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
        st.warning("⚠️ **Nuance Alert**: Counterfactual, conditional, or sarcastic subtext detected.")

    st.markdown("---")
    st.markdown("### 🔍 Sentence Clause Breakdown")
    for chunk in data["clause_breakdown"]:
        lab = chunk.get('sentiment', 'NEUTRAL').upper()
        badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
        conf = float(chunk.get('confidence', 0.5))
        
        with st.expander(f"Clause #{chunk.get('chunk_id', 1)}: {badge} ({conf*100:.1f}%)", expanded=True):
            st.write(f"**Text:** *\"{chunk.get('text', '')}\"*")
            st.progress(conf)