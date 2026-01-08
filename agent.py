from pprint import pprint
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

load_dotenv()

class TutorResponse(BaseModel):
    content: str
    interests: List[str]
    lexile_level: str

#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).with_structured_output(TutorResponse)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.1).with_structured_output(TutorResponse)

class State(TypedDict, total=False):
    name: str
    interests: List[str]
    lexile_level: str
    messages: Annotated[list, add_messages]

def tutor(state: State) -> State:
    name = state.get("name", "friend")
    interests = ", ".join(state.get("interests", []))
    lexile_level = state.get("lexile_level", "200L")
    
    prompt = f"""Your task is to identify the interests and evaluate the lexile level of {name} 
    You want to find what his interests are, these are what we previously know about him {interests}
    Check vocabulary and grammar on user input to infer the lexile level

    The content of your response should follow these lexile guidelines. 
    - 0-200: Very simple words, short sentences (5-8 words), concrete only
    - 200-500: Simple explanations, basic vocabulary
    - 500-800: Clear explanations, moderate sentence length
    - 800-1100: More complex vocabulary, longer explanations okay
    - 1101-: Advanced vocabulary and abstract concepts fine

    Personalize your response content with the user name {name} and relevant to {interests}

    Return:
    - content: your response to the user
    - lexile_level: estimate based on user responses
    - interests: append any interests inferred from user messages
    """

    messages = [SystemMessage(content=prompt)] + state.get("messages", [])[-10:]
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "interests": response.interests,
        "lexile_level": response.lexile_level
    }

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
        "messages": [HumanMessage(content="What is 6x7???? six-seveeen")]
    })
    pprint(result)
