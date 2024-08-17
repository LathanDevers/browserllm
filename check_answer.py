from haystack import Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever,ElasticsearchEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from components.lc_generator import LCGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
import json
import torch
import yaml

if __name__=="__main__":
    document_store=ElasticsearchDocumentStore(hosts="http://localhost:9200",index="sentence")
    pipeline=Pipeline()
    prompt_template1="""
                You are a personal assistant.
                Answer the question with the following context.
                
                Context:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                Question: {{query}}
                Answer:
            """
    prompt_template="""
                Tu es un assistant personnel.
                Réponds à la question en prenant le contexte en compte.
                
                Contexte:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                Question: {{query}}
                Réponse:
            """
    pipeline.add_component(name="query_embedder",instance=SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2",device=ComponentDevice.from_str("cuda:0")))
    pipeline.add_component(name="retriever",instance=ElasticsearchEmbeddingRetriever(document_store=document_store,top_k=10))
    pipeline.add_component(name="prompt_builder",instance=PromptBuilder(template=prompt_template))
    pipeline.add_component(name="generator",instance=LCGenerator())

    pipeline.connect(sender="query_embedder.embedding",receiver="retriever.query_embedding")
    pipeline.connect(sender="retriever.documents",receiver="prompt_builder.documents")
    pipeline.connect(sender="prompt_builder.prompt",receiver="generator.context")

    cuda_available=torch.cuda.is_available()
    print(f"cuda available : {cuda_available}")

    with open("./dataset/question_answers/question_answers.yml","r",encoding="utf-8") as yf:
        questions=yaml.safe_load(yf)
    qa_dict={}
    c=0
    for f in questions.keys():
        for i in range(5):
            c+=1
            j=i*2
            q=questions[f][j][f"question {i+1}"]
            try:
                a=questions[f][j+1][f"réponse {i+1}"]
            except:
                a=questions[f][j+1][f"reponse {i+1}"]
            try:
                qa_dict[f].update({q:a})
            except:
                qa_dict.update({f:{q:a}})
    
    correspondance_list={}
    for section in qa_dict.keys():
        for query in qa_dict[section].keys():
            results=pipeline.run(data={"query_embedder":{"text":query},"prompt_builder":{"query":query}})
            answer=results['generator']['answer']
            correspondance_list.update({qa_dict[section][query]:answer[0]})
            print({qa_dict[section][query]:answer[0]})
            with open("src/corresponding_answers.json","w",encoding="utf-8") as g:
                json.dump(correspondance_list,g,ensure_ascii=False)