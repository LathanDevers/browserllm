from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever,ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
import re
from haystack import component,Pipeline
from haystack.utils import ComponentDevice, Device
from typing import List

@component
class FlareIntegration:
    def __init__(self):
        self.generator=LlamaCppGenerator(model="mistral-7b-instruct-v0.2.Q4_K_M.gguf",n_ctx=32768,n_batch=128, model_kwargs={"n_gpu_layers": -1},generation_kwargs={"max_tokens": 1024,"temperature": 0.1})
        self.generator.warm_up()
        

        self.embedder=SentenceTransformersTextEmbedder(progress_bar=False,device=ComponentDevice.from_str("cuda:0"))
        self.embedder.warm_up()

        self.document_store=ElasticsearchDocumentStore(hosts="http://localhost:9200",index="passage")

        self.retriever=ElasticsearchEmbeddingRetriever(document_store=self.document_store)

    
    @component.output_types(answer=str)
    def run(self,query:str,loop:int):
        context="""You are a personal assistant.
        Given the Question, write a complete answer.
        Question :
        {{query}}
        Answer : """
        self.prompt_builder1=PromptBuilder(template=context)
        prompt1=self.prompt_builder1.run(query=query)
        first_answer=self.generator.run(prompt=f"[INST] {prompt1} [/INST]")["replies"][0]

        loop_answer=first_answer+" "
        answer=first_answer+" "

        prompt="""
            You are a personal assistant.
            Given the context, answer the question.
            Context :
            {% for d in documents %}
                {{d.content}}
            {% endfor %}
            Question :
            {{query}}
            """

        for _ in range(loop):

            embeddings=self.embedder.run(loop_answer)
            documents=self.retriever.run(query_embedding=embeddings["embedding"],top_k=3)

            self.prompt_builder2=PromptBuilder(template=prompt)
            prompt2=self.prompt_builder2.run(documents=documents["documents"],query=query)
            loop_answer=self.generator.run(prompt=f"[INST] {prompt2} [/INST]")["replies"][0]
            answer+=loop_answer+" "

        return {"answer":answer}



if __name__=="__main__":
    query="Quelle est l'objectif principal de la fondation 21st Century Tiger?"

    pipeline=Pipeline()

    pipeline.add_component(name="flare",instance=FlareIntegration())

    questions=pipeline.run(data={"flare":{"query":query,"loop":3}})


    print(questions)
