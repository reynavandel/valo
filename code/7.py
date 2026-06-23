from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small")

text = """Artificial Intelligence (AI) is a rapidly evolving field
of computer science focused on creating intelligent machines
capable of performing tasks that typically require human
intelligence."""

print("Original:\n", text)

s1 = summarizer(text, max_length=50, min_length=20, do_sample=False)
print("\nDefault:\n", s1[0]["summary_text"])