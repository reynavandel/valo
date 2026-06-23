from transformers import pipeline

analyzer = pipeline("sentiment-analysis")

def analyze(text):
    result = analyzer(text)[0]
    print(f"Text      : {text}")
    print(f"Sentiment : {result['label']} (Score: {result['score']:.4f})\n")

reviews = [
    "The product is amazing! I love it so much.",
    "I'm very disappointed. The service was terrible.",
    "It was an average experience, nothing special.",
    "Absolutely fantastic quality! Highly recommended.",
    "Not great, but not the worst either."
]

for review in reviews:
    analyze(review)