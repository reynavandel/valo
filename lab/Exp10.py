# ============================================================
# Program 10
# IPC Chatbot using LangChain + Cohere + Wikipedia
# ============================================================

# ------------------------------------------------------------
# Install Required Libraries
# ------------------------------------------------------------
!pip install langchain cohere wikipedia-api pydantic ipywidgets
!pip install langchain-community

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
from langchain import PromptTemplate, LLMChain
from langchain.llms import Cohere

from pydantic import BaseModel
from typing import Optional

import wikipediaapi
import getpass

# ------------------------------------------------------------
# Step 1: Enter Cohere API Key
# ------------------------------------------------------------
COHERE_API_KEY = getpass.getpass(
    "Enter your Cohere API Key: "
)

cohere_llm = Cohere(
    cohere_api_key=COHERE_API_KEY,
    model="command"
)

# ------------------------------------------------------------
# Step 2: Fetch IPC Summary from Wikipedia
# ------------------------------------------------------------
def fetch_ipc_summary():

    user_agent = "IPCChatbot/1.0 (contact: myemail@example.com)"

    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent=user_agent,
        language="en"
    )

    page = wiki_wiki.page("Indian Penal Code")

    if not page.exists():

        raise ValueError(
            "The Indian Penal Code page does not exist on Wikipedia."
        )

    return page.text[:5000]

ipc_content = fetch_ipc_summary()

# ------------------------------------------------------------
# Step 3: Pydantic Model
# ------------------------------------------------------------
class IPCResponse(BaseModel):

    section: Optional[str]

    explanation: Optional[str]

# ------------------------------------------------------------
# Step 4: Prompt Template
# ------------------------------------------------------------
prompt_template = PromptTemplate(

    input_variables=["question"],

    template="""
    You are a legal assistant chatbot specialized in the Indian Penal Code (IPC).

    Refer to the following IPC document content to answer the user's query.

    {ipc_content}

    User Question:

    {question}

    Provide a detailed answer,
    mentioning the relevant section if applicable.
    """

)

# ------------------------------------------------------------
# Step 5: Chatbot Function
# ------------------------------------------------------------
def get_ipc_response(question: str) -> IPCResponse:

    formatted_prompt = prompt_template.format(

        ipc_content=ipc_content,

        question=question

    )

    response = cohere_llm.predict(formatted_prompt)

    if "Section" in response:

        section = response.split("Section")[1].split(":")[0].strip()

        explanation = response.split(":",1)[-1].strip()

    else:

        section = None

        explanation = response.strip()

    return IPCResponse(

        section=section,

        explanation=explanation

    )

# ------------------------------------------------------------
# Step 6: Interactive Chatbot
# ------------------------------------------------------------
from IPython.display import display
import ipywidgets as widgets

def display_response(response: IPCResponse):

    print(
        f"Section: {response.section if response.section else 'N/A'}"
    )

    print(
        f"Explanation: {response.explanation}"
    )

def on_button_click(b):

    user_question = text_box.value

    try:

        response = get_ipc_response(user_question)

        display_response(response)

    except Exception as e:

        print(f"Error: {e}")

# ------------------------------------------------------------
# Create Widgets
# ------------------------------------------------------------
text_box = widgets.Text(

    value="",

    placeholder="Ask about the Indian Penal Code",

    description="You:",

    disabled=False

)

button = widgets.Button(

    description="Ask",

    disabled=False,

    button_style="",

    tooltip="Click to ask a question about IPC",

    icon="legal"

)

button.on_click(on_button_click)

# ------------------------------------------------------------
# Display Chatbot
# ------------------------------------------------------------
display(text_box, button)