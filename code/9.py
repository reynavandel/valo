# pip install wikipedia-api pydantic
import re
import wikipediaapi
from pydantic import BaseModel

class InstitutionDetails(BaseModel):
    founder: str | None
    founded: str | None
    branches: str | None
    employees: str | None
    summary: str | None

def fetch(name):
    wiki = wikipediaapi.Wikipedia(
        user_agent="InstitutionFetcher/1.0 (sohan@example.com)",
        language="en"
    )

    page = wiki.page(name)
    if not page.exists():
        print("Not found")
        return

    text = page.text

    def get(p):
        m = re.search(p, text, re.I)
        return m.group(1) if m else None

    data = InstitutionDetails(
        founder = get(r'by .*?([A-Z][a-z]+ [A-Z][a-z]+)'),
        founded = get(r'(\d{4})'),
        branches = "Yes" if "campus" in text.lower() else None,
        employees = get(r'([\d,]+)\s+(staff|employees)'),
        summary = " ".join(page.summary.split(". ")[:4])
    )

    #or u can simply print as : print(data.model_dump())
    print("\n--- Institution Details ---")
    print("Founder   :", data.founder)
    print("Founded   :", data.founded)
    print("Branches  :", data.branches)
    print("Employees :", data.employees)
    print("Summary   :", data.summary)

fetch(input("Enter Institution Name: "))