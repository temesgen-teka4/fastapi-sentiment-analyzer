# ==========================================
# FASTAPI SENTIMENT ANALYZER SERVICE (v3.2)
# ==========================================
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(
    title="AI Sentiment Analyzer API",
    description="Production-grade NLP REST API featuring advanced RoBERTa analysis, clause chunking, and intent guardrails.",
    version="3.2.0",
)

# Load the advanced RoBERTa sentiment model on server startup
print("Initializing advanced RoBERTa AI model pipeline...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# Request schema using Pydantic for input validation
class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        example="Initially, I heard negative reviews, but after trying it out myself, I am completely blown away!",
    )


def process_sentiment_analysis(text: str) -> dict:
    """Core logic: intent guardrails, clause chunking, three-way sentiment scoring, and contrastive overriding."""
    clean_input = text.strip().strip('"').strip("'")

    # 🛡️ SMART INTENT GUARDRAIL: Catch informational questions / confusion statements
    inquiry_patterns = [
        "confused about", "confused by", "what is", "how does", 
        "can you explain", "tell me about", "why is", "what are", "who is"
    ]
    
    is_inquiry = any(pattern in clean_input.lower() for pattern in inquiry_patterns)
    
    if is_inquiry:
        return {
            "total_chunks": 1,
            "overall_sentiment": "NEUTRAL",
            "average_score": 0.0,
            "positive_chunks": 0,
            "negative_chunks": 0,
            "neutral_chunks": 1,
            "applied_recency_override": False,
            "breakdown": [{
                "chunk_id": 1,
                "text": clean_input,
                "sentiment": "NEUTRAL",
                "confidence": 0.95
            }]
        }

    # Split by clause boundaries and contrastive conjunctions
    raw_chunks = re.split(
        r"(?<=[.!?])\s+|,\s+(?=but|however|although|yet)",
        clean_input,
        flags=re.IGNORECASE,
    )
    sentences = [c.strip() for c in raw_chunks if c.strip()]

    if not sentences:
        raise ValueError("No valid text clauses found for processing.")

    chunk_results = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for i, sentence in enumerate(sentences):
        prediction = sentiment_pipeline(sentence[:512])[0]
        label = prediction["label"].upper()
        confidence = prediction["score"]

        if label == "POSITIVE":
            positive_count += 1
        elif label == "NEGATIVE":
            negative_count += 1
        else:
            neutral_count += 1

        chunk_results.append(
            {
                "chunk_id": i + 1,
                "text": sentence,
                "sentiment": label,
                "confidence": round(confidence, 4),
            }
        )

    def label_to_score(sentiment: str, conf: float) -> float:
        if sentiment == "POSITIVE":
            return conf
        elif sentiment == "NEGATIVE":
            return -conf
        else:
            return 0.0

    # Contrastive overriding logic
    if len(chunk_results) > 1 and any(
        word in clean_input.lower()
        for word in ["but", "however", "although", "yet"]
    ):
        final_chunk = chunk_results[-1]
        overall_sentiment = final_chunk["sentiment"]
        average_score = label_to_score(overall_sentiment, final_chunk["confidence"])
        applied_override = True
    else:
        total_score = sum(
            label_to_score(c["sentiment"], c["confidence"]) for c in chunk_results
        )
        average_score = total_score / len(chunk_results)
        
        if average_score > 0.15:
            overall_sentiment = "POSITIVE"
        elif average_score < -0.15:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"
            
        applied_override = False

    return {
        "total_chunks": len(sentences),
        "overall_sentiment": overall_sentiment,
        "average_score": round(average_score, 4),
        "positive_chunks": positive_count,
        "negative_chunks": negative_count,
        "neutral_chunks": neutral_count,
        "applied_recency_override": applied_override,
        "breakdown": chunk_results,
    }


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "AI Sentiment Analyzer API",
        "version": "3.2.0",
    }


@app.post("/analyze")
def analyze_text_endpoint(payload: TextAnalysisRequest):
    try:
        results = process_sentiment_analysis(payload.text)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))