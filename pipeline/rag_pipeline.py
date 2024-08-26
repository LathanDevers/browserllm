from components.hypothetical_document_embedder import HypotheticalDocumentEmbedder
from components.lc_generator import LCGenerator
from components.query_rewriter import QueryRewriter
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever,ElasticsearchEmbeddingRetriever
from components.summarizer import Summarizer
from haystack import Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.builders import AnswerBuilder
from haystack.components.rankers import LostInTheMiddleRanker,SentenceTransformersDiversityRanker
from haystack.utils import ComponentDevice

class RagPipeline:
    """
        pre_retrieval_options : "rewrite", "expanse", "FLARE", "HyDE"\n
        retrieval_options : "embeddings", "bm25"\n
        post_retrieval_options : ["summary","fusion","rerank"]\n
        post_retrieval_type : \n
        generation_options : 
    """
    def __init__(self,pre_retrieval_options=None,retrieval_options="embeddings",post_retrieval_options=[],post_retrieval_type={},generation_options="mixtral",hosts="http://192.168.2.179:9200",index="sentence",top_k_retriever=10):
        self.pre_retrieval_options=pre_retrieval_options
        self.retrieval_options=retrieval_options
        self.post_retrieval_options=post_retrieval_options
        self.post_retrieval_type=post_retrieval_type
        self.generation_options=generation_options
        self.hosts=hosts
        self.index=index
        self.top_k_retriever=top_k_retriever

        self.data={}

        self.pipeline=Pipeline()

        self.rewriter=None
        self.expanser=None
        self.flare=None
        self.hyde=None

        self.from_query=None

        self.to_retriever=None

        self.query_embedder=None

        self.retrieval=None

        self.to_summarizer=None
        self.from_summarizer=None
        self.summarizer=None

        self.to_reranker=None
        self.from_reranker=None
        self.reranker=None

        self.to_merger=None
        self.from_merger=None
        self.merger=None

        self.generator=None

        self.answer=None

        self.document_store=ElasticsearchDocumentStore(hosts=self.hosts,index=self.index)

        if self.generation_options=="mixtral":
            self.model_name="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

        self.create_pre_retrieval_components()
        self.create_retrieval_components()
        self.create_post_retrieval_components()
        self.create_generation_components()
        self.create_answer_component()

        self.pipeline.draw("./pipeline.png")

    def create_pre_retrieval_components(self):
        if self.pre_retrieval_options=="rewrite":
            self.rewriter=QueryRewriter()
            self.pipeline.add_component(name="rewriter",instance=self.rewriter)

            self.from_query="rewriter.rewrited_query"
        elif self.pre_retrieval_options=="expanse":
            pass
        elif self.pre_retrieval_options=="flare":
            pass
        elif self.pre_retrieval_options=="hyde":
            self.hyde=HypotheticalDocumentEmbedder()
            self.pipeline.add_component(name="hyde",instance=self.hyde)

    def create_retrieval_components(self):
        if self.retrieval_options=="embeddings":
            self.query_embedder=SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2",device=ComponentDevice.from_str("cuda:0"))

            self.pipeline.add_component(name="query_embedder",instance=self.query_embedder)
            
            if self.pre_retrieval_options!="":
                self.pipeline.connect(sender=self.from_query,receiver="query_embedder.text")

            self.from_query_embedder="query_embedder.embedding"

            self.retrieval=ElasticsearchEmbeddingRetriever(document_store=self.document_store,top_k=self.top_k_retriever)
            self.pipeline.add_component(name="retrieval",instance=self.retrieval)

            self.to_retriever="query_embedder.text"

            self.pipeline.connect(sender="query_embedder.embedding",receiver="retrieval.query_embedding")
        elif self.retrieval_options=="bm25":
            self.retrieval=ElasticsearchBM25Retriever(document_store=self.document_store,top_k=self.top_k_retriever)
            self.pipeline.add_component(name="retrieval",instance=self.retrieval)

            self.to_retriever="retrieval.query"

            if self.pre_retrieval_options!=None:
                self.pipeline.connect(sender=self.from_query,receiver="retrieval.query")

        self.from_retriever="retrieval.documents"

    def create_post_retrieval_components(self):
        self.from_post_retrieval=self.from_retriever
        if "summary" in self.post_retrieval_options:
            self.summarizer=Summarizer(model_name=self.model_name)
            self.pipeline.add_component(name="summarizer",instance=self.summarizer)

            self.to_summarizer="summarizer.documents"
            self.from_summarizer="summarizer.summarized_documents"

            self.from_post_retrieval=self.from_summarizer
        if "fusion" in self.post_retrieval_options:
            self.from_post_retrieval=self.from_merger
        if "rerank" in self.post_retrieval_options:
            self.to_reranker="reranker.documents"
            self.from_reranker="reranker.documents"

            self.from_post_retrieval=self.from_reranker

            if self.post_retrieval_type["rerank"]=="Lost In The Middle":
                self.reranker=LostInTheMiddleRanker(top_k=self.top_k_retriever)
            elif self.post_retrieval_type["rerank"]=="Diversity":
                self.reranker=SentenceTransformersDiversityRanker(top_k=self.top_k_retriever)
            self.pipeline.add_component(name="reranker",instance=self.reranker)


        """if self.to_summarizer!=None:
            self.pipeline.connect(sender=self.from_retriever,receiver=self.to_summarizer)
            if self.to_merger!=None:
                self.pipeline.connect(sender=self.from_summarizer,receiver=self.to_merger)
                if self.to_reranker!=None:
                    self.pipeline.connect(sender=self.from_merger,receiver=self.to_reranker)"""
        
        if self.to_summarizer!=None:
            self.pipeline.connect(sender=self.from_retriever,receiver=self.to_summarizer)
            if self.to_reranker!=None:
                self.pipeline.connect(sender=self.from_summarizer,receiver=self.to_reranker)


    def create_generation_components(self):
        prompt_template="""
                You are a personal assistant.
                Aswer the question with the following context.
                
                Context:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                Question: {{query}}
                Answer:
            """
        self.generator_prompt_builder=PromptBuilder(template=prompt_template)
        self.pipeline.add_component(name="generator_prompt_builder",instance=self.generator_prompt_builder)

        self.to_generator_prompt_builder="generator_prompt_builder.documents"

        self.pipeline.connect(sender=self.from_post_retrieval,receiver=self.to_generator_prompt_builder)

        self.create_llamacpp_generator()

        self.pipeline.add_component(name="generator",instance=self.generator)

        self.pipeline.connect(sender="generator_prompt_builder.prompt",receiver="generator.context")

    def create_llamacpp_generator(self):
        self.generator=LCGenerator(model_name=self.model_name)
    
    def create_answer_component(self):
        self.answer_builder=AnswerBuilder()
        self.pipeline.add_component(name="answer_builder",instance=self.answer_builder)

        self.pipeline.connect(sender="generator.answer",receiver="answer_builder.replies")
        if "summary" in self.post_retrieval_options:
            self.pipeline.connect(sender=self.from_summarizer,receiver="answer_builder.documents")
        else:
            self.pipeline.connect(sender="retrieval",receiver="answer_builder.documents")
    
    def run(self,query:str):
        data={}
        if self.pre_retrieval_options=="rewrite":
            data.update({"rewriter":{"query":query}})
        
        if self.retrieval_options=="embeddings":
            data.update({"query_embedder":{"text":query}})
        elif self.retrieval_options=="bm25":
            data.update({"retrieval":{"query":query}})
        
        if "summary" in self.post_retrieval_options:
            data.update({"summarizer":{"query":query}})
        
        data.update({"answer_builder":{"query":query}})
            
        result=self.pipeline.run(data=data)
        return {"answer":result["answer_builder"]["answers"][0].data,"documents":result["answer_builder"]["answers"][0].documents}