import gensim.downloader as api

wv = api.load("glove-wiki-gigaword-50")

seed = input("Enter a seed word: ")

try:
    sw = [w for w, _ in wv.most_similar(seed, topn=5)]
    print("Similar words:", sw)

    para = " ".join([
        f"The {seed} was surrounded by {sw[0]} and {sw[1]}.",
        f"People often associate {seed} with {sw[2]} and {sw[3]}.",
        f"In the land of {seed}, {sw[4]} was a common sight.",
    ])

    print("\nGenerated Paragraph:\n", para)

except KeyError:
    print("Word not found")