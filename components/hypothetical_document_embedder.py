from typing import List
from numpy import array,mean
from haystack import component,Document,Pipeline
from components.lc_generator import LCGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

@component
class HypotheticalDocumentEmbedder:

    @component.output_types(hypothetical_embedding=List[float])
    def run(self,documents:List[Document]):
        stacked_embeddings=array([doc.embedding for doc in documents])
        avg_embeddings=mean(stacked_embeddings,axis=0)
        hyde_vector=avg_embeddings.reshape((1,len(avg_embeddings)))
        return {"hypothetical_embedding":hyde_vector[0].tolist()}

if __name__=="__main__":

    generator=LCGenerator()
    prompt_builder=PromptBuilder(template="""Given a question, generate a paragraph of text that answers the question.    Question: {{query}}    Paragraph:""")
    adapter=OutputAdapter("{{asnwers | build_doc}}",output_type=List[Document],custom_filters={"build_doc":lambda data:[Document(content=d) for d in data]})
    embedder=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2",progress_bar=False)
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

    query="What should I do if I have a fever ?"
    results=pipeline.run(data={"prompt_builder":{"query":query}})
    print(results)