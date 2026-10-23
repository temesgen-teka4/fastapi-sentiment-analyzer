# ==========================================
# STREAMLIT SENTIMENT ANALYZER DASHBOARD (v3.2)
# ==========================================
import re
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer v3.2",
    page_icon="🤖",
    layout="centered"
)

# Cache the model loading
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

with st.spinner("Loading advanced RoBERTa AI model weights... Please wait..."):
    sentiment_pipeline = load_sentiment_model()

# Header Section
st.title("🤖 AI Text Intelligence Dashboard")
st.caption("Powered by RoBERTa Transformers • Featuring Clause-Level Smart Aggregation & Intent Guardrails")

st.markdown("---")

# User Text Input
default_text = "I am a bit confused about the government policies, but overall I want to understand how they work."
user_text = st.text_area("Paste your input text below:", value=default_text, height=140)

if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        clean_input = user_text.strip().strip('"').strip("'")

        # 🛡️ SMART INTENT GUARDRAIL
        inquiry_patterns = ["confused about", "confused by", "what is", "how does", "can you explain", "tell me about", "why is", "what are", "who is"]
        is_inquiry = any(pattern in clean_input.lower() for pattern in inquiry_patterns)

        if is_inquiry:
            chunk_results = [{
                "chunk_id": 1,
                "text": clean_input,
                "sentiment": "NEUTRAL",
                "confidence": 0.95
            }]
            overall_sentiment = "NEUTRAL"
            overall_score = 0.95
            positive_count = 0
            negative_count = 0
            neutral_count = 1
            applied_override = False
            sentences = [clean_input]
        else:
            # Split text into clauses/sentences using regex
            raw_chunks = re.split(r'(?<=[.!?])\s+|,\s+(?=but|however|although|yet)', clean_input, flags=re.IGNORECASE)
            sentences = [c.strip() for c in raw_chunks if c.strip()]

            chunk_results = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for i, sentence in enumerate(sentences):
                prediction = sentiment_pipeline(sentence[:512])[0]
                label = prediction['label'].upper()
                confidence = prediction['score']

                if label == "POSITIVE":
                    positive_count += 1
                elif label == "NEGATIVE":
                    negative_count += 1
                else:
                    neutral_count += 1

                chunk_results.append({
                    "chunk_id": i + 1,
                    "text": sentence,
                    "sentiment": label,
                    "confidence": confidence
                })

            def label_to_score(sentiment: str, conf: float) -> float:
                if sentiment == "POSITIVE":
                    return conf
                elif sentiment == "NEGATIVE":
                    return -conf
                else:
                    return 0.0

            has_pivot = any(word in clean_input.lower() for word in ['but', 'however', 'although', 'yet'])
            if len(chunk_results) > 1 and has_pivot:
                final_chunk = chunk_results[-1]
                overall_sentiment = final_chunk['sentiment']
                overall_score = final_chunk['confidence']
                applied_override = True
            else:
                total_score = sum(label_to_score(c['sentiment'], c['confidence']) for c in chunk_results)
                avg = total_score / len(chunk_results)
                
                if avg > 0.15:
                    overall_sentiment = "POSITIVE"
                elif avg < -0.15:
                    overall_sentiment = "NEGATIVE"
                else:
                    overall_sentiment = "NEUTRAL"
                    
                overall_score = abs(avg)
                applied_override = False

        # Save to session state
        st.session_state["analysis_data"] = {
            "overall_sentiment": overall_sentiment,
            "overall_score": overall_score,
            "total_chunks": len(sentences),
            "positive_chunks": positive_count,
            "negative_chunks": negative_count,
            "neutral_chunks": neutral_count,
            "applied_override": applied_override,
            "is_inquiry": is_inquiry,
            "chunk_results": chunk_results
        }

# --- RENDER RESULTS & INTERACTIVE LOOP ---
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    
    st.markdown("### 📊 Overall Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_emojis = {"POSITIVE": "POSITIVE 😃", "NEGATIVE": "NEGATIVE 🙁", "NEUTRAL": "NEUTRAL 😐"}
        st.metric("Overall Sentiment", sentiment_emojis.get(data["overall_sentiment"], data["overall_sentiment"]))

    with col2:
        st.metric("Confidence Score", f"{data['overall_score'] * 100:.1f}%")

    with col3:
        st.metric("Total Clauses Analyzed", data["total_chunks"])

    if data.get("is_inquiry"):
        st.info("🛡️ **Intent Guardrail Triggered**: Detected information-seeking or inquiry phrasing. Categorized securely as Neutral.")
    elif data["applied_override"]:
        st.info("💡 **AI Insight Applied**: Detected contrastive pivot (*'but' / 'however'*). Priority weighting applied.")

    st.markdown("---")
    st.markdown("### 🔍 Sentence Clause Breakdown")

    for chunk in data["chunk_results"]:
        lab = chunk['sentiment']
        badge = "🟢 POSITIVE" if lab == "POSITIVE" else ("🔴 NEGATIVE" if lab == "NEGATIVE" else "🟡 NEUTRAL")
        
        with st.expander(f"Clause #{chunk['chunk_id']}: {badge} ({chunk['confidence']*100:.1f}%)", expanded=True):
            st.write(f"**Text:** *\"{chunk['text']}\"*")
            st.progress(chunk['confidence'])

    # --- INTERACTIVE NEXT-STEP FOLLOW-UP LOOP ---
    st.markdown("---")
    st.markdown("### 🔄 Interactive Follow-Up Actions")
    st.write("What would you like to explore next with this analyzed text?")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📊 Show Clause Counts"):
            st.info(f"🟢 Positive: {data['positive_chunks']} | 🔴 Negative: {data['negative_chunks']} | 🟡 Neutral: {data['neutral_chunks']}")
            
    with action_col2:
        if st.button("📝 Summarize Intent"):
            st.success(f"AI Summary: The text maintains a dominant **{data['overall_sentiment']}** tone across {data['total_chunks']} evaluated semantic clause(s).")
            
    with action_col3:
        if st.button("✨ Clear Results"):
            del st.session_state["analysis_data"]
            st.rerun()