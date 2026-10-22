# ==========================================
# FASTAPI SENTIMENT ANALYZER SERVICE
# ==========================================
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(
    title="AI Sentiment Analyzer API",
    description="Production-grade NLP REST API featuring clause chunking and contrastive intent overriding.",
    version="3.0.0",
)

# Load Hugging Face pipeline on server startup
print("Initializing AI model pipeline...")
sentiment_pipeline = pipeline("sentiment-analysis")


# Request schema using Pydantic for input validation
class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        example="Initially, I heard negative reviews, but after trying it out myself, I am completely blown away!",
    )


def process_sentiment_analysis(text: str) -> dict:
    """Core logic: clause chunking, sentiment scoring, and contrastive overriding."""
    clean_input = text.strip().strip('"').strip("'")

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

    for i, sentence in enumerate(sentences):
        prediction = sentiment_pipeline(sentence[:512])[0]
        label = prediction["label"]
        confidence = prediction["score"]

        if label == "POSITIVE":
            positive_count += 1
        else:
            negative_count += 1

        chunk_results.append(
            {
                "chunk_id": i + 1,
                "text": sentence,
                "sentiment": label,
                "confidence": round(confidence, 4),
            }
        )

    # Contrastive overriding logic
    if len(chunk_results) > 1 and any(
        word in clean_input.lower()
        for word in ["but", "however", "although", "yet"]
    ):
        final_chunk = chunk_results[-1]
        overall_sentiment = final_chunk["sentiment"]
        average_score = (
            final_chunk["confidence"]
            if overall_sentiment == "POSITIVE"
            else -final_chunk["confidence"]
        )
        applied_override = True
    else:
        total_score = sum(
            c["confidence"] if c["sentiment"] == "POSITIVE" else -c["confidence"]
            for c in chunk_results
        )
        average_score = total_score / len(chunk_results)
        overall_sentiment = "POSITIVE" if average_score >= 0 else "NEGATIVE"
        applied_override = False

    return {
        "total_chunks": len(sentences),
        "overall_sentiment": overall_sentiment,
        "average_score": round(average_score, 4),
        "positive_chunks": positive_count,
        "negative_chunks": negative_count,
        "applied_recency_override": applied_override,
        "breakdown": chunk_results,
    }


@app.get("/")
def health_check():
    """Health check endpoint to confirm API service status."""
    return {
        "status": "online",
        "service": "AI Sentiment Analyzer API",
        "version": "3.0.0",
    }


@app.post("/analyze")
def analyze_text_endpoint(payload: TextAnalysisRequest):
    """Analyze long text inputs using model chunking and sentiment override logic."""
    try:
        results = process_sentiment_analysis(payload.text)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))