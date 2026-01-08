### Friction log

## Documentation on evaluate-graph example

On the Langsmith [evaluate-graph documentation](https://docs.langchain.com/langsmith/evaluate-graph): 

- Bug, Attempted to create dataset using [code snippet](https://docs.langchain.com/langsmith/evaluate-graph#create-a-dataset)
    ``` 
    ls_client = Client()
    dataset = ls_client.create_dataset(
        name="weather agent",
        inputs=[{"question": q} for q in questions],
        outputs=[{"answers": a} for a in answers],
    )
    ```
    I tested using langsmith 0.61 Results in error:
    ```
    evaluate_graph/dataset.py", line 18, in <module> dataset = ls_client.create_dataset( ^^^^^^^^^^^^^^^^^^^^^^^^^ TypeError: Client.create_dataset() got an unexpected keyword argument 'inputs'
    ```
    Reviewing the docs I found a [working example of sdk dataset creation](https://docs.langchain.com/langsmith/manage-datasets-programmatically#create-a-dataset-from-list-of-values) That works.
    Fix: Update example code snippet to include create_examples separately as sdk docs show above.

- Suggestion, it would be nice a hint box that recommends using structured output, when evaluating outputs, I had some evaluation runs that gave me inconsistent values due to additional markdown in the output. 
