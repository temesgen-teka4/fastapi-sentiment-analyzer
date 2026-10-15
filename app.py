# ==========================================
# 1. IMPORTS
# We import the 'pipeline' function from Hugging Face's transformers library.
# ==========================================
from transformers import pipeline

# ==========================================
# 2. LOAD PRE-TRAINED MODEL
# We initialize an open-source NLP sentiment model.
# The first run will download model weights (~250MB) locally.
# ==========================================
print("Loading AI model... Please wait...")
sentiment_pipeline = pipeline("sentiment-analysis")


def text_analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment using a pre-trained Transformer model.
    Returns label (POSITIVE/NEGATIVE) and confidence score.
    """
    # The pipeline returns a list containing a dict: [{'label': 'POSITIVE', 'score': 0.998}]
    raw_results = sentiment_pipeline(text)
    prediction = raw_results[0]

    return {
        "input": text,
        "sentiment": prediction["label"],
        "confidence": round(prediction["score"], 4)
    }


# ==========================================
# 3. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n-- AI SENTIMENT ANALYZER v2.0 (Transformer Powered) --")
    
    user_input = input("Enter a sentence to analyze: ")

    if user_input.strip():
        result = text_analyze_sentiment(user_input)
        print("\n-- Analysis Result --")
        print(f"Text:       {result['input']}")
        print(f"Sentiment:  {result['sentiment']}")
        print(f"Confidence: {result['confidence'] * 100}%")
        print("------------------------")
    else:
        print("No text provided. Exiting program.")