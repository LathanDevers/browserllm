from haystack import component,Pipeline
from typing import List
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice

@component
class MultiEmbedder:
    def __init__(self):
        self.query_embedder=SentenceTransformersTextEmbedder(progress_bar=False,device=ComponentDevice.from_str("cuda:0"))
        self.query_embedder.warm_up()
    
    @component.output_types(embeddings=List[float])
    def run(self,queries:List[str]):
        embeddings=[]

        for query in queries:
            results=self.query_embedder.run(text=query)
            embeddings.append(results['embedding'])

        return {"embeddings":embeddings}

if __name__=="__main__":
    pipeline=Pipeline()

    pipeline.add_component(name="embedder",instance=MultiEmbedder())

    results=pipeline.run(data={"embedder":{"queries":["a","b","c","d"]}})
    
    print(results)