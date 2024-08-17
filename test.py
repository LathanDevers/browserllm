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
from plot import plot_i
import json
import math

def create_retrieval_pipeline(document_store,top_k=10,t=1):
    pipeline=Pipeline()

    if t==1:#normal bm25
        retriever=ElasticsearchBM25Retriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="retriever",instance=retriever)
    elif t==2:#normal embedd
        query_embedder=SentenceTransformersTextEmbedder(progress_bar=False,device=ComponentDevice.from_str("cuda:0"))
        retriever=ElasticsearchEmbeddingRetriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="query_embedder",instance=query_embedder)
        pipeline.add_component(name="retriever",instance=retriever)
        pipeline.connect(sender="query_embedder.embedding",receiver="retriever.query_embedding")
    elif t==3:#rewrite + embedd
        query_embedder=SentenceTransformersTextEmbedder(progress_bar=False,device=ComponentDevice.from_str("cuda:0"))
        retriever=ElasticsearchEmbeddingRetriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="query_embedder",instance=query_embedder)
        pipeline.add_component(name="retriever",instance=retriever)
        pipeline.connect(sender="query_embedder.embedding",receiver="retriever.query_embedding")
    elif t==4:#expand + embedd
        query_embedder=SentenceTransformersTextEmbedder(progress_bar=False,device=ComponentDevice.from_str("cuda:0"))
        retriever=ElasticsearchEmbeddingRetriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="query_embedder",instance=query_embedder)
        pipeline.add_component(name="retriever",instance=retriever)
        pipeline.connect(sender="query_embedder.embedding",receiver="retriever.query_embedding")
    elif t==5:#flare+embedd
        query_embedder=MultiEmbedder()
        retriever=EmbeddingMultiRetriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="query_embedder",instance=query_embedder)
        pipeline.add_component(name="retriever",instance=retriever)
        pipeline.connect(sender="query_embedder.embeddings",receiver="retriever.embeddings")
    elif t==6:#hyde+embedd
        retriever=ElasticsearchEmbeddingRetriever(document_store=document_store,top_k=top_k)
        pipeline.add_component(name="retriever",instance=retriever)


    return pipeline

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

    with open("src/rewrited_queries.json",'r',encoding="utf-8") as rqf:
        rewrited_queries=json.load(rqf)
    with open("src/expanded_queries.json",'r',encoding="utf-8") as eqf:
        expanded_queries=json.load(eqf)
    with open("src/flare_queries.json",'r',encoding="utf-8") as fqf:
        alt_queries_flare=json.load(fqf)
    with open("src/docs_hyde.json",'r',encoding="utf-8") as dhf:
        docs_hyde=json.load(dhf)
    
    indexes=["sentence","passage"]
    di=0
    percentplus=0
    
    for t in tqdm([1,2,3,4],desc="pipelines",unit="pipelines"):
        tp_list=[]
        precisions=[]
        rappels=[]
        mrrs=[]
        tp=0
        tp_pres=0
        l=len(qa_dict.keys())*5 if t!=6 else 100
        document_store=ElasticsearchDocumentStore(hosts="http://localhost:9200",index=indexes[di])
        for i in tqdm(range(10,310,50),desc=f"# retrieved documents with pipeline {t}",leave=False,unit="fichiers récupérés",):
            pipeline=create_retrieval_pipeline(document_store=document_store,top_k=i,t=t)
            tp=0
            mrr=[]
            precision=[]
            rappel=[]
            for fichier in tqdm(qa_dict.keys(), desc=f"fichiers with # {i}",leave=False,unit="fichiers"):
                for question in tqdm(qa_dict[fichier],leave=False,desc="questions",unit="questions"):
                    if t==1:
                        results=pipeline.run(data={"retriever":{"query":question}})
                    elif t==2:
                        results=pipeline.run(data={"query_embedder":{"text":question}})
                    elif t==3:
                        results=pipeline.run(data={"query_embedder":{"text":rewrited_queries[question]}})
                    elif t==4:
                        results=pipeline.run(data={"query_embedder":{"text":expanded_queries[question]}})
                    elif t==5:
                        a=alt_queries_flare[question]
                        a.append(question)
                        results=pipeline.run(data={"query_embedder":{"queries":a}})
                    elif t==6:
                        if question in docs_hyde.keys():
                            results=pipeline.run(data={"retriever":{"query_embedding":docs_hyde[question]}})
                        else:
                            results=None
                    
                    if results!=None:
                        d=[r.meta["title"].replace(" — Wikipédia","") for r in results["retriever"]['documents']]
                        if fichier in d:
                            tp+=1
                            mrr.append(1/(d.index(fichier)+1))
                            precision.append(sum([f==fichier for f in d])/i)
                        else:
                            mrr.append(0)
                            precision.append(0)

            tp_list.append(tp/l)
            mrrs.append(mrr)
            precisions.append(precision)

            #precision = tp/(tp+fp)=tp/i

            #rappel = tp/(tp+fn)=>tp={0,1} pas utile

            #mrr = 1/index(fichier)

            tp_pres=tp

        with open(f"src/tests/tp_list{t}{indexes[di]}.json","w") as tp_list_f:
            json.dump(tp_list,tp_list_f)
        with open(f"src/tests/mrrs{t}{indexes[di]}.json","w") as mrrs_f:
            json.dump(mrrs,mrrs_f)
        with open(f"src/tests/precisions{t}{indexes[di]}.json","w") as precision_f:
            json.dump(precisions,precision_f)

    #plot_i(title=f"pipeline : {indexes[d]} -> embeddings",tp_list=tp_list)