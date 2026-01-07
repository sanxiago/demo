from agent import agent
from pprint import pprint

state = {}
while True:
    user_input = input("Type message: ")
    if user_input.lower() == "quit":
        break
    
    state["user_message"] = user_input
    state = agent.invoke(state)
    pprint(state) 
