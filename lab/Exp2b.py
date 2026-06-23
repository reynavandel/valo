# ============================================================
# Experiment 2(b)
# Domain Specific Word Embeddings
# ============================================================

# -----------------------------
# Install Libraries
# -----------------------------
!pip install gensim scikit-learn matplotlib

# -----------------------------
# Import Libraries
# -----------------------------
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Load Word2Vec Model
# -----------------------------
print("Loading pre-trained word vectors...")

word_vectors = api.load("word2vec-google-news-300")

print("Model Loaded Successfully!")

# -----------------------------
# Technology Domain Words
# -----------------------------
domain_words = [
    "computer",
    "software",
    "hardware",
    "algorithm",
    "data",
    "network",
    "programming",
    "machine",
    "learning",
    "artificial"
]

domain_vectors = np.array([
    word_vectors[word]
    for word in domain_words
])

# -----------------------------
# Visualization Function
# -----------------------------
def visualize_word_embeddings(
        words,
        vectors,
        method="pca",
        perplexity=5
):

    if method == "pca":
        reducer = PCA(n_components=2)

    else:
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity
        )

    reduced_vectors = reducer.fit_transform(vectors)

    plt.figure(figsize=(10,8))

    for i, word in enumerate(words):

        plt.scatter(
            reduced_vectors[i,0],
            reduced_vectors[i,1]
        )

        plt.text(
            reduced_vectors[i,0]+0.02,
            reduced_vectors[i,1]+0.02,
            word
        )

    plt.title(f"Visualization using {method.upper()}")
    plt.grid(True)
    plt.show()

# PCA Visualization
visualize_word_embeddings(
    domain_words,
    domain_vectors,
    method="pca"
)

# t-SNE Visualization
visualize_word_embeddings(
    domain_words,
    domain_vectors,
    method="tsne",
    perplexity=3
)

# -----------------------------
# Find Similar Words
# -----------------------------
def generate_similar_words(word):

    try:

        similar_words = word_vectors.most_similar(
            word,
            topn=5
        )

        print(f"\nTop 5 Similar Words to '{word}'")

        for similar_word, similarity in similar_words:

            print(f"{similar_word} : {similarity:.4f}")

    except KeyError:

        print("Word not found in vocabulary.")

# Examples
generate_similar_words("computer")
generate_similar_words("learning")