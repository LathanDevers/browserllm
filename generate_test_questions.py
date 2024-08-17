import yaml
from tqdm import tqdm
from haystack import Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever,ElasticsearchEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice, Device
from components.flare import FlareIntegration
from components.multi_retriever import EmbeddingMultiRetriever
from haystack.components.joiners import BranchJoiner
from components.multi_embedder import MultiEmbedder
import torch
from components.query_rewriter import QueryRewriter
from components.query_expander import QueryExtander
from plot import plot_i
import json
from typing import List
from numpy import array,mean
from haystack import component,Document,Pipeline
from components.lc_generator import LCGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from components.hypothetical_document_embedder import HypotheticalDocumentEmbedder



if __name__=="__main__":
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
    
    print(f"Nombre de questions : {c}")

    q_dict=qa_dict.keys()

    rewrited_queries={}
    expanded_queries={}
    alt_queries_flare={}
    docs_hyde={}

    r1=input("generate rewrite queries ? (y/n) : ")
    r2=input("generate expanded queries ? (y/n) : ")
    r3=input("generate flare queries ? (y/n) : ")
    r4=input("generate hyde queries ? (y/n) : ")
    ############################################# rewrited queries #############################################################
    if r1.lower() == "y":
        rewrite_pipeline=Pipeline()
        rewriter=QueryRewriter()
        rewrite_pipeline.add_component(name="rewriter",instance=rewriter)

        for fichier in tqdm(qa_dict.keys(), desc="rewrite questions"):
            for question in tqdm(qa_dict[fichier],leave=False):
                rewrited_query=rewrite_pipeline.run(data={"rewriter":{"query":question}})
                rewrited_queries.update({question:rewrited_query["rewriter"]["rewrited_query"]})
        with open("src/rewrited_queries.json","w",encoding="utf-8") as rqf:
            json.dump(rewrited_queries,rqf,ensure_ascii=False)
    ############################################################################################################################

    ############################################# expanded queries #############################################################
    if r2.lower() == "y":
        expand_pipeline=Pipeline()
        expander=QueryExtander()
        expand_pipeline.add_component(name="expander",instance=expander)

        for fichier in tqdm(qa_dict.keys(), desc="expand questions"):
            for question in tqdm(qa_dict[fichier],leave=False):
                expanded_query=expand_pipeline.run(data={"expander":{"query":question}})
                expanded_queries.update({question:expanded_query["expander"]["extanded_query"]})
        with open("src/expanded_queries.json","w",encoding="utf-8") as eqf:
            json.dump(expanded_queries,eqf,ensure_ascii=False)
    ############################################################################################################################


    ############################################# flare queries ################################################################
    if r3.lower()=="y":
        pipeline=Pipeline()
        pipeline.add_component(name="flare",instance=FlareIntegration())

        for fichier in tqdm([k for k in qa_dict.keys()][:5], desc="flare questions"):
            for question in tqdm(qa_dict[fichier],leave=False):
                answer=pipeline.run(data={"flare":{"query":question,"loop":1}})
                alt_queries_flare.update({question:answer["flare"]["answer"]})
        with open("src/flare_queries.json","w",encoding="utf-8") as fqf:
            json.dump(alt_queries_flare,fqf,ensure_ascii=False)
    ############################################################################################################################


    ############################################# hyde documents ###############################################################
    if r4.lower()=="y":
        generator=LCGenerator()
        prompt_builder=PromptBuilder(template="""Given a question, generate a paragraph of text that answers the question.    Question: {{query}}    Paragraph:""")
        adapter=OutputAdapter("{{asnwers | build_doc}}",output_type=List[Document],custom_filters={"build_doc":lambda data:[Document(content=d) for d in data]})
        embedder=SentenceTransformersDocumentEmbedder()
        hyde=HypotheticalDocumentEmbedder()

        pipeline=Pipeline()

        pipeline.add_component(name="prompt_builder",instance=prompt_builder)
        pipeline.add_component(name="generator",instance=generator)
        pipeline.add_component(name="adapter",instance=adapter)
        pipeline.add_component(name="embedder",instance=embedder)
        pipeline.add_component(name="hyde",instance=hyde)

        pipeline.connect("prompt_builder","generator")
        pipeline.connect("generator.answer","adapter.asnwers")
        pipeline.connect("adapter.output","embedder.documents")
        pipeline.connect("embedder.documents","hyde.documents")

        for fichier in tqdm([k for k in qa_dict.keys()][:20], desc="hyde questions"):
            for question in tqdm(qa_dict[fichier],leave=False):
                results=pipeline.run(data={"prompt_builder":{"query":question}})
                docs_hyde.update({question:results["hyde"]["hypothetical_embedding"]})
        with open("src/docs_hyde.json","w",encoding="utf-8") as dhf:
            json.dump(docs_hyde,dhf,ensure_ascii=False)
    ############################################################################################################################