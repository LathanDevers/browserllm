from haystack import component,Pipeline,Document
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever,ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from components.flare import FlareIntegration
from typing import Set,List

@component
class EmbeddingMultiRetriever:
    def __init__(self,document_store,top_k):
        self.document_store=document_store
        self.top_k=top_k
        self.retriever=ElasticsearchEmbeddingRetriever(document_store=self.document_store,top_k=self.top_k)
    
    @component.output_types(documents=List[Document])
    def run(self,embeddings:List[float]):
        retrieved_documents=[]
        
        self.top_k==max(3,self.top_k/len(embeddings))

        for query_embedding in embeddings:
            results=self.retriever.run(query_embedding=query_embedding,top_k=self.top_k)
            retrieved_documents+=results["documents"]

        return {"documents":retrieved_documents}
        



if __name__=="__main__":
    document_store=ElasticsearchDocumentStore(hosts="http://localhost:9200",index="passage")
    
    flare=FlareIntegration()
    retriever=EmbeddingMultiRetriever(document_store=document_store,top_k=10)

    pipeline=Pipeline()

    pipeline.add_component(name="flare",instance=flare)
    pipeline.add_component(name="retriever",instance=retriever)

    pipeline.connect(sender="flare.queries",receiver="retriever.queries")

    results=pipeline.run(data={"flare":{"query":"What is a cat ?"}})
    print(results)
