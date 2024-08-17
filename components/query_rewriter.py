from haystack import component
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack import Pipeline

@component
class QueryRewriter:
    """
    A component rewriting the query
    """
    def __init__(self):
        self.generator=LlamaCppGenerator(model="mistral-7b-instruct-v0.2.Q4_K_M.gguf",n_ctx=32768,n_batch=128, model_kwargs={"n_gpu_layers": -1},generation_kwargs={"temperature": 0.1})
        self.generator.warm_up()

    @component.output_types(rewrited_query=str)
    def run(self,query:str):
        self.instruct = f"""You are a personal assistant.
        Rewrite the question to add some better context.
        Question: {query}
        Rewrited question: """
        output = self.generator.run(prompt=f"[INST] {self.instruct} [/INST]",generation_kwargs={"max_tokens": 1024})
        response=output["replies"][0]
        return {"rewrited_query":response}


if __name__=="__main__":
    pipeline=Pipeline()
    pipeline.add_component(instance=QueryRewriter(),name="query_rewriter")
    results=pipeline.run({"query_rewriter":{"query":"Où et quand les bars à chats ont-ils été popularisés pour la première fois?"}})
    print(results["query_rewriter"]["rewrited_query"])