import gensim.downloader as api
from transformers import pipeline

wv        = api.load("glove-wiki-gigaword-100")
generator = pipeline("text-generation", model="gpt2")

def enrich_prompt(prompt, keyword):
    similar = wv.most_similar(keyword, topn=1)[0][0]
    print(f"Replacing '{keyword}' -> '{similar}'")
    return prompt.replace(keyword, similar)

def generate(prompt):
    return generator(prompt, max_length=100)[0]["generated_text"]

original_prompt = "Who is king."
enriched_prompt = enrich_prompt(original_prompt, "king")

original_response = generate(original_prompt)
enriched_response = generate(enriched_prompt)

print(f"\nOriginal  : {original_response}")
print(f"Enriched  : {enriched_response}")


