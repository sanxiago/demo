import json
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Inputs(BaseModel):
    name: str
    interests: list[str]
    lexile_level: str
    user_message: str

class TestCase(BaseModel):
    inputs: Inputs

class Dataset(BaseModel):
    items: list[TestCase]

llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.8).with_structured_output(Dataset)
llm_ollama = ChatOllama(model="qwen2.5:7b", temperature=0.8).with_structured_output(Dataset)

prompt = """Generate 20 test use cases for a K-12 tutor.

name: user name (include unicode characters)
interests: array of user interests
lexile_level: literacy score (BR-200L beginner to 1200L proficient)
user_message: user prompt (grammar, spelling, length correlate to lexile_level)

Include multiple languages, typos, and grammatical errors for lower lexile levels."""

openai_response = llm_openai.invoke(prompt)
ollama_response = llm_ollama.invoke(prompt)

with open("dataset.jsonl", "w") as f:
    for item in openai_response.items + ollama_response.items:
        f.write(json.dumps({"inputs": item.inputs.model_dump()}) + "\n")

print("Saved to dataset.jsonl")
