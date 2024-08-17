from haystack import component
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack import Pipeline
from typing import List

@component
class LCGenerator:
    def __init__(self, model_name="mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        self.generator=LlamaCppGenerator(model=model_name,n_ctx=32768,n_batch=128, model_kwargs={"n_gpu_layers": 32},generation_kwargs={"temperature": 0.1})
        self.generator.warm_up()

    @component.output_types(answer=List[str])
    def run(self,context:str):
        output=self.generator.run(prompt=f"[INST] {context} [/INST]",generation_kwargs={"max_tokens": 1024})
        
        response=output["replies"]
        
        return {"answer":response}

if __name__=="__main__":
    pipeline=Pipeline()

    pipeline.add_component(name="generator",instance=LCGenerator())

    results=pipeline.run(data={"generator":{"context":"What should I do if I have a fever ?"}})

    print(results)