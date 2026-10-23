# ==========================================
# FASTAPI SENTIMENT ANALYZER SERVICE (v4.0)
# ==========================================
import os
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(
    title="AI Text Intelligence API",
    description="Production-grade NLP REST API featuring LLM Deep Reasoning & RoBERTa Clause Analysis.",
    version="4.0.0",
)

# Load lightweight fallback RoBERTa model
print("Initializing RoBERTa fallback pipeline...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Initialize Gemini Client if API key exists
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        example="If the support team were actually responsive, this would easily be a 5-star product.",
    )


def analyze_with_llm_reasoning(text: str) -> dict:
    """Uses Gemini LLM for human-level intent extraction, sarcasm detection, and sentiment analysis."""
    prompt = f"""
Analyze the following text with deep human-level comprehension. Account for sarcasm, double negatives, counterfactual conditionals, and implicit subtext.

Text: "{text}"

Return ONLY a raw JSON object with this exact structure:
{{
    "overall_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence_score": float between 0.0 and 1.0,
    "is_sarcastic_or_conditional": boolean,
    "detected_nuance": "Brief label describing tone (e.g. Counterfactual Criticism, Heavy Sarcasm, Direct Praise)",
    "human_reasoning": "A concise 1-2 sentence explanation of what the speaker ACTUALLY means in reality."
}}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(response.text)


def process_sentiment_analysis(text: str) -> dict:
    clean_input = text.strip().strip('"').strip("'")

    # Clause splitting for structural breakdown
    raw_chunks = re.split(r"(?<=[.!?])\s+|,\s+(?=but|however|although|yet)", clean_input, flags=re.IGNORECASE)
    sentences = [c.strip() for c in raw_chunks if c.strip()]

    chunk_results = []
    for i, sentence in enumerate(sentences):
        pred = sentiment_pipeline(sentence[:512])[0]
        chunk_results.append({
            "chunk_id": i + 1,
            "text": sentence,
            "sentiment": pred["label"].upper(),
            "confidence": round(pred["score"], 4)
        })

    # Execute Deep Reasoning Engine if API Key is available
    if client:
        try:
            llm_result = analyze_with_llm_reasoning(clean_input)
            return {
                "engine_used": "Gemini 2.5 Flash Deep Reasoning",
                "total_chunks": len(sentences),
                "overall_sentiment": llm_result["overall_sentiment"].upper(),
                "confidence_score": round(llm_result["confidence_score"], 4),
                "is_sarcastic_or_conditional": llm_result["is_sarcastic_or_conditional"],
                "detected_nuance": llm_result["detected_nuance"],
                "human_reasoning": llm_result["human_reasoning"],
                "clause_breakdown": chunk_results
            }
        except Exception as e:
            print(f"LLM Engine error, falling back to RoBERTa: {e}")

    # Fallback RoBERTa Logic
    total_score = sum(c["confidence"] if c["sentiment"] == "POSITIVE" else (-c["confidence"] if c["sentiment"] == "NEGATIVE" else 0.0) for c in chunk_results)
    avg_score = total_score / len(chunk_results) if chunk_results else 0.0
    overall = "POSITIVE" if avg_score > 0.15 else ("NEGATIVE" if avg_score < -0.15 else "NEUTRAL")

    return {
        "engine_used": "RoBERTa Baseline Pipeline",
        "total_chunks": len(sentences),
        "overall_sentiment": overall,
        "confidence_score": round(abs(avg_score), 4),
        "is_sarcastic_or_conditional": False,
        "detected_nuance": "Standard Pattern Analysis",
        "human_reasoning": "Baseline statistical classification executed without LLM contextual reasoning.",
        "clause_breakdown": chunk_results
    }


@app.get("/")
def health_check():
    return {"status": "online", "service": "AI Text Intelligence API", "version": "4.0.0"}


@app.post("/analyze")
def analyze_text_endpoint(payload: TextAnalysisRequest):
    try:
        results = process_sentiment_analysis(payload.text)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))