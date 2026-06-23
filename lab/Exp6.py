# ============================================================
# Program 6
# Sentiment Analysis using Hugging Face Pre-trained Model
# ============================================================

# ------------------------------------------------------------
# Install Required Libraries
# ------------------------------------------------------------
!pip install transformers torch

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
from transformers import pipeline

# ------------------------------------------------------------
# Load Sentiment Analysis Pipeline
# ------------------------------------------------------------
print("Loading Sentiment Analysis Model...")

sentiment_analyzer = pipeline("sentiment-analysis")

# ------------------------------------------------------------
# Function to Analyze Sentiment
# ------------------------------------------------------------
def analyze_sentiment(text):

    """
    Analyze the sentiment of a given text.
    """

    result = sentiment_analyzer(text)[0]

    label = result["label"]

    score = result["score"]

    print(f"\nInput Text: {text}")

    print(f"Sentiment: {label} (Confidence: {score:.4f})\n")

    return result

# ------------------------------------------------------------
# Example: Customer Reviews
# ------------------------------------------------------------
customer_reviews = [

    "The product is amazing! I love it so much.",

    "I'm very disappointed. The service was terrible.",

    "It was an average experience, nothing special.",

    "Absolutely fantastic quality! Highly recommended.",

    "Not great, but not the worst either."

]

# ------------------------------------------------------------
# Analyze Sentiment
# ------------------------------------------------------------
print("\nCustomer Sentiment Analysis Results:")

for review in customer_reviews:

    analyze_sentiment(review)