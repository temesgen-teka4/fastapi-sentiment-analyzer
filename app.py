import sys  # allows us to interact with system arguments.

def text_analyze_sentiment(text: str) -> dict:
    # Clean the input text by converting it to lowercase
    cleaned_text = text.lower()

    # Simple keyword rule lists (keep them lowercase so matching works!)
    positive_words = ["good", "happy", "smile", "great", "python"]
    negative_words = ["sad", "crying", "terrible", "hate", "slow", "hard"]

    # Calculate word score 
    score = 0
    words = cleaned_text.split()

    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1

    # Determine the sentiment result based on the calculated score
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral" 
       
    return {
        "input": text,
        "sentiment": label,
        "score": score
    }

if __name__ == "__main__":
    print("-- AI SENTIMENT ANALYZER v1.0 --")
    
    # Prompt the user to enter input
    user_input = input("Enter a sentence to analyze: ")

    if user_input.strip():
        # Make sure the function name matches your def statement!
        result = text_analyze_sentiment(user_input)
        print("\n-- Analyzed Result --")
        print(f"Text: {result['input']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']}")
        print("------------------------")
    else:
        print("No text provided. Exiting program!!")