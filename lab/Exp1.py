# ============================================================
# Experiment 1
# Explore Pre-trained Word Vectors & Vector Arithmetic
# ============================================================

# ------------------------------------------------------------
# Step 1: Install required libraries
# ------------------------------------------------------------
!pip install gensim numpy

# ------------------------------------------------------------
# Step 2: Import required libraries
# ------------------------------------------------------------
import gensim.downloader as api
import numpy as np
from numpy.linalg import norm

# ------------------------------------------------------------
# Step 3: Load Pre-trained Word2Vec Model
# (This downloads the model only once. It may take a few
# minutes because the model is around 1.6 GB.)
# ------------------------------------------------------------
print("Loading pre-trained word vectors...")

word_vectors = api.load("word2vec-google-news-300")

print("Model loaded successfully!")

# ------------------------------------------------------------
# Step 4: Function to perform Vector Arithmetic
# Example:
# king - man + woman ≈ queen
# ------------------------------------------------------------
def explore_word_relationships(word1, word2, word3):

    try:

        # Get vectors of the input words
        vec1 = word_vectors[word1]
        vec2 = word_vectors[word2]
        vec3 = word_vectors[word3]

        # Perform vector arithmetic
        result_vector = vec1 - vec2 + vec3

        # Find the most similar words
        similar_words = word_vectors.similar_by_vector(
            result_vector,
            topn=10
        )

        # Remove input words from output
        input_words = {word1, word2, word3}

        filtered_words = [
            (word, similarity)
            for word, similarity in similar_words
            if word not in input_words
        ]

        print("\n---------------------------------------")
        print(f"Word Relationship: {word1} - {word2} + {word3}")
        print("---------------------------------------")

        print("Top Similar Words:")

        for word, similarity in filtered_words[:5]:
            print(f"{word} : {similarity:.4f}")

    except KeyError as e:
        print(f"Error: {e} not found in vocabulary.")


# ------------------------------------------------------------
# Step 5: Test Vector Arithmetic
# ------------------------------------------------------------
explore_word_relationships("king", "man", "woman")

explore_word_relationships("paris", "france", "germany")

explore_word_relationships("apple", "fruit", "carrot")


# ------------------------------------------------------------
# Step 6: Function to Calculate Cosine Similarity
# ------------------------------------------------------------
def analyze_similarity(word1, word2):

    try:

        similarity = word_vectors.similarity(word1, word2)

        print(f"\nSimilarity between '{word1}' and '{word2}' = {similarity:.4f}")

    except KeyError as e:
        print(f"Error: {e} not found in vocabulary.")


# ------------------------------------------------------------
# Step 7: Test Similarity
# ------------------------------------------------------------
analyze_similarity("cat", "dog")

analyze_similarity("computer", "keyboard")

analyze_similarity("music", "art")


# ------------------------------------------------------------
# Step 8: Function to Find Similar Words
# ------------------------------------------------------------
def find_most_similar(word):

    try:

        similar_words = word_vectors.most_similar(
            word,
            topn=5
        )

        print(f"\nMost Similar Words to '{word}':")

        for similar_word, similarity in similar_words:

            print(f"{similar_word} : {similarity:.4f}")

    except KeyError as e:

        print(f"Error: {e} not found in vocabulary.")


# ------------------------------------------------------------
# Step 9: Find Similar Words
# ------------------------------------------------------------
find_most_similar("happy")

find_most_similar("sad")

find_most_similar("technology")