# ==========================================
# STREAMLIT SENTIMENT ANALYZER DASHBOARD
# ==========================================
import re
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer v3.0",
    page_icon="🤖",
    layout="centered"
)

# Cache the model loading so it stays fast across interactions
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

with st.spinner("Loading AI model weights... Please wait..."):
    sentiment_pipeline = load_sentiment_model()

# Header Section
st.title("🤖 AI Document Sentiment Analyzer")
st.caption("Powered by DistilBERT Transformers • Featuring Clause-Level Smart Aggregation & Contrastive Override")

st.markdown("---")

# User Text Input
default_text = "Initially, I heard some negative feedback about this tool online and was worried, but after trying it out myself, I am completely blown away by how amazing it works!"
user_text = st.text_area("Paste your input text below:", value=default_text, height=140)

if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        clean_input = user_text.strip().strip('"').strip("'")

        # Split text into clauses/sentences using regex
        raw_chunks = re.split(r'(?<=[.!?])\s+|,\s+(?=but|however|although|yet)', clean_input, flags=re.IGNORECASE)
        sentences = [c.strip() for c in raw_chunks if c.strip()]

        chunk_results = []
        positive_count = 0
        negative_count = 0

        for i, sentence in enumerate(sentences):
            prediction = sentiment_pipeline(sentence[:512])[0]
            label = prediction['label']
            confidence = prediction['score']

            if label == "POSITIVE":
                positive_count += 1
            else:
                negative_count += 1

            chunk_results.append({
                "chunk_id": i + 1,
                "text": sentence,
                "sentiment": label,
                "confidence": confidence
            })

        # Contrastive overriding logic
        has_pivot = any(word in clean_input.lower() for word in ['but', 'however', 'although', 'yet'])
        if len(chunk_results) > 1 and has_pivot:
            final_chunk = chunk_results[-1]
            overall_sentiment = final_chunk['sentiment']
            overall_score = final_chunk['confidence']
            applied_override = True
        else:
            total_score = sum(c['confidence'] if c['sentiment'] == "POSITIVE" else -c['confidence'] for c in chunk_results)
            avg = total_score / len(chunk_results)
            overall_sentiment = "POSITIVE" if avg >= 0 else "NEGATIVE"
            overall_score = abs(avg)
            applied_override = False

        st.markdown("### 📊 Overall Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if overall_sentiment == "POSITIVE":
                st.metric("Overall Sentiment", "POSITIVE 😃")
            else:
                st.metric("Overall Sentiment", "NEGATIVE 🙁")

        with col2:
            st.metric("Confidence Score", f"{overall_score * 100:.1f}%")

        with col3:
            st.metric("Total Clauses Analyzed", len(sentences))

        if applied_override:
            st.info("💡 **AI Insight Applied**: Detected contrastive pivot (*'but' / 'however'*). The final clause received priority weighting to capture the true human intent.")

        st.markdown("---")
        st.markdown("### 🔍 Sentence Breakdown")

        for chunk in chunk_results:
            is_pos = chunk['sentiment'] == "POSITIVE"
            badge = "🟢 POSITIVE" if is_pos else "🔴 NEGATIVE"
            
            with st.expander(f"Clause #{chunk['chunk_id']}: {badge} ({chunk['confidence']*100:.1f}%)", expanded=True):
                st.write(f"**Text:** *\"{chunk['text']}\"*")
                st.progress(chunk['confidence'])