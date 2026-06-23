# pip install gensim scikit-learn matplotlib
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wv = api.load("glove-wiki-gigaword-100")

# Visualize using PCA
def visualize(words):
    vecs   = [wv[w] for w in words]
    points = PCA(n_components=2).fit_transform(vecs)
    plt.scatter(points[:, 0], points[:, 1])
    for i, w in enumerate(words):
        plt.text(points[i, 0], points[i, 1], w)
    plt.title("PCA - Word Embeddings")
    plt.show()

def word_arithmetic(w1, w2, w3):
    print(f"\n{w1} - {w2} + {w3} =")
    for w, s in wv.most_similar(positive=[w1, w3], negative=[w2], topn=5):
        print(f"  {w}: {s:.4f}")

word_arithmetic("king", "man", "woman")

words = ["computer","laptop","keyboard","mouse","software",
         "hardware","internet","network","data","programming"]
visualize(words)
