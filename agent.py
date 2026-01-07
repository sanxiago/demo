import json
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

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
    - BR-200L: Very simple words, short sentences (5-8 words), concrete only
    - 200-500L: Simple explanations, basic vocabulary
    - 500-800L: Clear explanations, moderate sentence length
    - 800-1100L: More complex vocabulary, longer explanations okay
    - 1100L+: Advanced vocabulary and abstract concepts fine

    Personalize your response content with the user name {name}.

    The format of your response should be JSON formatted:
    1. content: the response content intended for the user
    2. lexile_level: based on their input and comprehension of previous messages.
    3. interests: This is an updated list of the user interests, append any new interests identified.

    Response should be in raw JSON no additional markdown or encoding.
    """

    messages = [SystemMessage(content=prompt)] + state.get("messages", [])
    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        return {
            "messages": [AIMessage(content=result["content"])],
            "interests": result["interests"],
            "lexile_level": result["lexile_level"]
        }
    except json.JSONDecodeError:
        return {
            "messages": [AIMessage(content=response.content)]
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
    print(result["messages"][-1].content)
