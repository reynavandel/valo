import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wv = api.load("glove-wiki-gigaword-100")

words = ["computer", "software", "hardware", "algorithm", "data",
         "network", "programming", "machine", "learning", "artificial"]

def visualize(words):
    vecs   = [wv[w] for w in words]
    points = PCA(n_components=2).fit_transform(vecs)
    plt.scatter(points[:, 0], points[:, 1])
    for i, w in enumerate(words):
        plt.text(points[i, 0], points[i, 1], w)
    plt.title("PCA - Technology Word Embeddings")
    plt.show()

def similar_words(word):
    print(f"\nTop 5 similar to '{word}':")
    for w, s in wv.most_similar(word, topn=5):
        print(f"  {w}: {s:.4f}")

visualize(words)
similar_words("computer")
similar_words("learning")