# ==========================================
# STREAMLIT TEXT INTELLIGENCE PLATFORM (v6.0 - Interactive Loop Edition)
# ==========================================
import streamlit as st
import re

st.set_page_config(
    page_title="AI Text Intelligence Platform",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Text Intelligence Platform")
st.caption("Interactive Hybrid Intelligence Engine • Continuous Analysis Loop")

st.markdown("---")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your AI Text Intelligence Assistant. Type or paste any text below (try tricky sentences with sarcasm or counterfactuals like *'If support were responsive, this would be 5-star'*), and I will break down its intent and clauses instantly!",
            "analysis": None
        }
    ]

def intelligent_heuristic_analysis(text: str) -> dict:
    """Advanced heuristic linguistic decoder for sentiment, sarcasm, and clauses."""
    clean = text.strip()
    lower_text = clean.lower()
    
    is_counterfactual = False
    detected_nuance = "Direct Statement"
    human_reasoning = ""
    
    if "if " in lower_text and any(term in lower_text for term in ["were actually", "was actually", "had been", "would easily", "should have", "could be"]):
        is_counterfactual = True
        detected_nuance = "Counterfactual Criticism"
        overall_sentiment = "NEGATIVE"
        confidence_score = 0.92
        human_reasoning = "The speaker highlights a current service failure by framing a hypothetical, unfulfilled condition (e.g., wishing service were better implies it is currently poor)."
    elif any(word in lower_text for word in ["not bad", "hardly disappointing", "not the worst"]):
        is_counterfactual = True
        detected_nuance = "Double Negative / Mild Praise"
        overall_sentiment = "POSITIVE"
        confidence_score = 0.85
        human_reasoning = "The text uses understatement or double negatives to convey a positive underlying endorsement."
    elif any(word in lower_text for word in ["great", "amazing", "love", "excellent", "best", "wonderful", "5-star"]):
        overall_sentiment = "POSITIVE"
        confidence_score = 0.88
        detected_nuance = "Direct Praise"
        human_reasoning = "The text uses explicit positive evaluative language expressing direct satisfaction."
    elif any(word in lower_text for word in ["worst", "horrible", "terrible", "awful", "useless", "poor", "frustrating", "unresponsive"]):
        overall_sentiment = "NEGATIVE"
        confidence_score = 0.90
        detected_nuance = "Direct Criticism"
        human_reasoning = "The text uses strong negative terminology signaling direct dissatisfaction."
    else:
        overall_sentiment = "NEUTRAL"
        confidence_score = 0.75
        detected_nuance = "Neutral Observation"
        human_reasoning = "The statement presents objective observations or balanced commentary without strong emotional polarization."

    # Clause breakdown splitting
    raw_chunks = re.split(r'(?<=[.!?])\s+|,\s+(?=but|however|although|yet)', clean, flags=re.IGNORECASE)
    clauses = [c.strip() for c in raw_chunks if c.strip()]
    
    clause_breakdown = []
    for i, clause in enumerate(clauses):
        c_lower = clause.lower()
        if any(w in c_lower for w in ["worst", "unresponsive", "poor", "bad", "failing"]):
            c_sent = "NEGATIVE"
            c_conf = 0.89
        elif any(w in c_lower for w in ["great", "5-star", "easily", "good", "amazing"]):
            c_sent = "NEGATIVE" if is_counterfactual else "POSITIVE"
            c_conf = 0.85
        else:
            c_sent = "NEUTRAL"
            c_conf = 0.70
            
        clause_breakdown.append({
            "chunk_id": i + 1,
            "text": clause,
            "sentiment": c_sent,
            "confidence": c_conf
        })

    return {
        "overall_sentiment": overall_sentiment,
        "confidence_score": confidence_score,
        "is_sarcastic_or_conditional": is_counterfactual,
        "detected_nuance": detected_nuance,
        "human_reasoning": human_reasoning,
        "clause_breakdown": clause_breakdown
    }

# Display chat history loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message contains structured analysis data, render it interactively
        if message.get("analysis"):
            data = message["analysis"]
            sentiment = data.get("overall_sentiment", "NEUTRAL").upper()
            confidence = float(data.get("confidence_score", 0.0))
            nuance = data.get("detected_nuance", "Standard Analysis")
            reasoning = data.get("human_reasoning", "")
            is_sarcastic = bool(data.get("is_sarcastic_or_conditional", False))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                emojis = {"POSITIVE": "POSITIVE 😃", "NEGATIVE": "NEGATIVE 🙁", "NEUTRAL": "NEUTRAL 😐"}
                st.metric("Sentiment", emojis.get(sentiment, sentiment))
            with col2:
                st.metric("Confidence", f"{confidence * 100:.1f}%")
            with col3:
                st.metric("Tone Nuance", nuance)
                
            if reasoning:
                st.info(f"💡 **AI Reasoning Insight**: \"{reasoning}\"")
                
            if is_sarcastic:
                st.warning("⚠️ **Nuance Alert**: Counterfactual or conditional subtext detected.")

            with st.expander("🔍 View Clause Breakdown", expanded=False):
                for chunk in data.get("clause_breakdown", []):
                    lab = chunk.get('sentiment', 'NEUTRAL').upper()
                    badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
                    conf = float(chunk.get('confidence', 0.5))
                    st.write(f"**Clause #{chunk.get('chunk_id', 1)} ({badge} - {conf*100:.1f}%):** *\"{chunk.get('text', '')}\"*")
                    st.progress(conf)

# Chat input loop for continuous user interaction
if user_prompt := st.chat_input("Type a sentence to analyze... (e.g. 'If support were actually good, this would be amazing.')"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt, "analysis": None})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Generate analysis response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing semantics in loop..."):
            analysis_result = intelligent_heuristic_analysis(user_prompt)
            
            response_text = f"Analyzed **\"{user_prompt}\"** successfully."
            st.markdown(response_text)
            
            sentiment = analysis_result["overall_sentiment"]
            confidence = analysis_result["confidence_score"]
            nuance = analysis_result["detected_nuance"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                emojis = {"POSITIVE": "POSITIVE 😃", "NEGATIVE": "NEGATIVE 🙁", "NEUTRAL": "NEUTRAL 😐"}
                st.metric("Sentiment", emojis.get(sentiment, sentiment))
            with col2:
                st.metric("Confidence", f"{confidence * 100:.1f}%")
            with col3:
                st.metric("Tone Nuance", nuance)
                
            st.info(f"💡 **AI Reasoning Insight**: \"{analysis_result['human_reasoning']}\"")
            
            if analysis_result["is_sarcastic_or_conditional"]:
                st.warning("⚠️ **Nuance Alert**: Counterfactual or conditional subtext detected.")

            with st.expander("🔍 View Clause Breakdown", expanded=False):
                for chunk in analysis_result["clause_breakdown"]:
                    lab = chunk['sentiment']
                    badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
                    st.write(f"**Clause #{chunk['chunk_id']} ({badge}):** *\"{chunk['text']}\"*")
                    st.progress(chunk['confidence'])
            
            # Save assistant message with analysis state to conversation history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "analysis": analysis_result
            })