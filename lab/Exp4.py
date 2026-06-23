# ============================================================
# Program 4
# Improve GenAI Prompts using Word Embeddings
# ============================================================

# ------------------------------------------------------------
# Install Required Libraries
# ------------------------------------------------------------
!pip install gensim transformers torch nltk

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import gensim.downloader as api
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

# Download tokenizer
nltk.download('punkt')

# ------------------------------------------------------------
# Load Pre-trained Word Vectors (GloVe)
# ------------------------------------------------------------
print("Loading pre-trained word vectors...")

word_vectors = api.load("glove-wiki-gigaword-100")

# ------------------------------------------------------------
# Function to Replace Keyword with Similar Word
# ------------------------------------------------------------
def replace_keyword_in_prompt(prompt, keyword, word_vectors, topn=1):

    """
    Replace only the specified keyword in the prompt
    with its most similar word.
    """

    words = word_tokenize(prompt)

    enriched_words = []

    for word in words:

        cleaned_word = word.lower().strip(string.punctuation)

        if cleaned_word == keyword.lower():

            try:

                similar_words = word_vectors.most_similar(
                    cleaned_word,
                    topn=topn
                )

                if similar_words:

                    replacement_word = similar_words[0][0]

                    print(f"Replacing '{word}' → '{replacement_word}'")

                    enriched_words.append(replacement_word)

                    continue

            except KeyError:

                print(f"'{keyword}' not found in vocabulary.")

        enriched_words.append(word)

    enriched_prompt = " ".join(enriched_words)

    print(f"\nEnriched Prompt: {enriched_prompt}")

    return enriched_prompt

# ------------------------------------------------------------
# Load GPT-2 Model
# ------------------------------------------------------------
print("\nLoading GPT-2 model...")

generator = pipeline(
    "text-generation",
    model="gpt2"
)

# ------------------------------------------------------------
# Function to Generate AI Response
# ------------------------------------------------------------
def generate_response(prompt, max_length=100):

    try:

        response = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1
        )

        return response[0]["generated_text"]

    except Exception as e:

        print(f"Error generating response: {e}")

        return None

# ------------------------------------------------------------
# Original Prompt
# ------------------------------------------------------------
original_prompt = "Who is king."

print(f"\nOriginal Prompt: {original_prompt}")

# Keyword
key_term = "king"

# ------------------------------------------------------------
# Enrich Prompt
# ------------------------------------------------------------
enriched_prompt = replace_keyword_in_prompt(
    original_prompt,
    key_term,
    word_vectors
)

# ------------------------------------------------------------
# Generate Response for Original Prompt
# ------------------------------------------------------------
print("\nGenerating response for the original prompt...")

original_response = generate_response(original_prompt)

print("\nOriginal Prompt Response:")

print(original_response)

# ------------------------------------------------------------
# Generate Response for Enriched Prompt
# ------------------------------------------------------------
print("\nGenerating response for the enriched prompt...")

enriched_response = generate_response(enriched_prompt)

print("\nEnriched Prompt Response:")

print(enriched_response)

# ------------------------------------------------------------
# Compare Responses
# ------------------------------------------------------------
print("\nComparison of Responses:")

print("\nOriginal Prompt Response Length:",
      len(original_response))

print("Enriched Prompt Response Length:",
      len(enriched_response))

print("\nOriginal Prompt Response Detail:",
      original_response.count("."))

print("Enriched Prompt Response Detail:",
      enriched_response.count("."))