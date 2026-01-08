import json
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent import agent

load_dotenv()

client = Client()

# Create dataset from JSONL file
def create_dataset(jsonl_path="dataset.jsonl", dataset_name="tutor-eval"):
    dataset = client.create_dataset(dataset_name)
    
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            client.create_example(inputs=data["inputs"], dataset_id=dataset.id)
    
    print(f"Created dataset: {dataset_name}")

# Target function - what we're evaluating
def target(inputs: dict) -> dict:
    result = agent.invoke({
        "name": inputs["name"],
        "interests": inputs.get("interests", []),
        "lexile_level": inputs.get("lexile_level", "500L"),
        "messages": [HumanMessage(content=inputs["user_message"])]
    })
    return {"response": result["messages"][-1].content}

# Evaluator - scores the response
def score_response(outputs: dict, inputs: dict) -> dict:
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Rate 1-5. Student: {inputs['name']} (Lexile: {inputs.get('lexile_level')})
Question: {inputs['user_message']}
Response: {outputs['response']}
Reply with only a number."""

    score = int(judge.invoke(prompt).content.strip())
    return {"score": score, "key": "quality"}

if __name__ == "__main__":
    try:
        create_dataset()  # Uncomment to create dataset first
    except Exception as e:
        print(e)
        pass
    results = evaluate(
        target,
        data="tutor-eval",
        evaluators=[score_response],
        experiment_prefix="tutor-test"
    )
    print(results)
