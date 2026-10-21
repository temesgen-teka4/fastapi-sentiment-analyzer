# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
import re
from transformers import pipeline

print("Loading AI model... Please wait...")
sentiment_pipeline = pipeline("sentiment-analysis")


def analyze_long_text(text: str) -> dict:
    """
    Analyzes text with intelligent clause weighting and contrastive overriding.
    """
    clean_input = text.strip().strip('"').strip("'")

    # Split by clause boundaries and contrastive conjunctions
    raw_chunks = re.split(r'(?<=[.!?])\s+|,\s+(?=but|however|although|yet)', clean_input, flags=re.IGNORECASE)
    sentences = [c.strip() for c in raw_chunks if c.strip()]
    
    if not sentences:
        return {"error": "No valid text chunks found."}

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
            "confidence": round(confidence, 4)
        })

    # --- INTELLIGENT AGGREGATION LOGIC ---
    if len(chunk_results) > 1 and any(word in clean_input.lower() for word in ['but', 'however', 'although', 'yet']):
        final_chunk = chunk_results[-1]
        overall_sentiment = final_chunk['sentiment']
        average_score = final_chunk['confidence'] if overall_sentiment == "POSITIVE" else -final_chunk['confidence']
        print("\n[AI Insight] Detected contrastive pivot. Applying recency override to capture true intent.")
    else:
        total_score = sum(c['confidence'] if c['sentiment'] == "POSITIVE" else -c['confidence'] for c in chunk_results)
        average_score = total_score / len(chunk_results)
        overall_sentiment = "POSITIVE" if average_score >= 0 else "NEGATIVE"

    return {
        "total_chunks": len(sentences),
        "overall_sentiment": overall_sentiment,
        "average_score": round(average_score, 4),
        "positive_chunks": positive_count,
        "negative_chunks": negative_count,
        "breakdown": chunk_results
    }


# ==========================================
# 2. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n-- AI DOCUMENT SENTIMENT ANALYZER v3.0 (Smart Chunking & Overriding) --")
    
    print("Paste your long text below (press Enter when done):")
    user_input = input("> ")

    if user_input.strip():
        result = analyze_long_text(user_input)
        
        print("\n==============================")
        print("      OVERALL SUMMARY         ")
        print("==============================")
        print(f"Total Sentences Analyzed: {result['total_chunks']}")
        print(f"Overall Document Sentiment: {result['overall_sentiment']}")
        print(f"Average Sentiment Score:   {result['average_score']}")
        print(f"Positive Chunks / Negative Chunks: {result['positive_chunks']} / {result['negative_chunks']}")
        print("------------------------------")
        
        print("\n--- Breakdown by Sentence ---")
        for chunk in result['breakdown']:
            print(f"[{chunk['chunk_id']}] {chunk['sentiment']} ({chunk['confidence']*100}%): {chunk['text']}")
    else:
        print("No text provided. Exiting program.")