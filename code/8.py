#pip -q install langchain langchain-community langchain-cohere cohere
#pip install requests==2.32.4

# from google.colab import drive
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
import getpass

# Mount Drive
# drive.mount('/content/drive')

# Load file
# text = open("/content/drive/MyDrive/Teaching.txt").read()

# ALTERNATE SAFE OPTION : Sample text (instead of file)
text = "Artificial Intelligence is transforming industries by enabling automation and data-driven decisions."

# API key
api_key = "1ra7Bir2NcHqTxVotCPaMiJVqive1Z40zNmqStZ4"

# Model
llm = ChatCohere(cohere_api_key=api_key, model="command-r-08-2024")

# Prompt template
prompt = PromptTemplate.from_template("""
    Summarize the text and give key points.

    Text: {text}

    Output:
    Summary:
    Key Points:
    Sentiment:
""")

# Run
res = llm.invoke(prompt.format(text=text))
print(res.content)