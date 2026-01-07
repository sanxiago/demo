from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

class State(TypedDict, total=False):
    name: str
    interests: List[str]
    lexile_level: str
    user_message: str
    response: str

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.1)

def tutor(state: State) -> State:
    name = state.get("name", "friend")
    interests = ", ".join(state.get("interests", []))
    lexile_level = state.get("lexile_level", "200L")
    user_message = state.get("user_message", "") 
    
    prompt = f""" Your task is to identify the interests and evaluate the lexile level of {name} 
    You want to find what his interests are, these are what we previously know about him {interests}
    Check vocabulary and grammar on user input to infer the lexile level

    The content of your response should follow these lexile guidelines. 
    - BR-200L: Very simple words, short sentences (5-8 words), concrete only
    - 200-500L: Simple explanations, basic vocabulary
    - 500-800L: Clear explanations, moderate sentence length
    - 800-1100L: More complex vocabulary, longer explanations okay
    - 1100L+: Advanced vocabulary and abstract concepts fine
    Shorter responses can be used when appropiate but you should avoid going beyond the known and inferred lexile level of the user.

Personalize your response content with the user name {name}.

The format of your response should be JSON formatted, with two items:
1. content:  the response content intended for the user
2. lexile_level: based on their input and comprehension of previous messages.
3. interests: This is an updated list of the user interests, append any new interests idenftified.

Response should be in raw JSON no additional markdown or encoding.

The user gave the following message: {user_message}

"""
    
    response = llm.invoke(prompt)
    return {"response": response.content}

graph = StateGraph(State)
graph.add_node("tutor", tutor)
graph.add_edge(START, "tutor")
graph.add_edge("tutor", END)
agent = graph.compile()

if __name__ == "__main__":
    result = agent.invoke({
        "name": "Alex",
        "interests": ["Legos", "dinosaurs", "space"],
        "lexile_level": "600L",
        "user_message": "What is 6x7???? six-seveeen"
    })
    print(result["response"])
