from pprint import pprint
from agent import agent, State
from langchain_core.messages import HumanMessage

name = input("What is your name? \n")
state: State = {
        "name": name,
        "messages": [],
        "interests": [], 
        "lexile_level": ""
        }

while True:
    user_input = input("Type Message> ")
    if user_input == "quit":
        break
    
    state["messages"].append(HumanMessage(content=user_input))
    state = agent.invoke(state)
    
    print(f"Tutor: {state['messages'][-1].content}\n")
    #print(f"Interests: {state['interests']}")
    #print(f"lexile_level: {state['lexile_level']}")
