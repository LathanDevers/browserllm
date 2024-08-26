from haystack import component,Document
from typing import List
from components.lc_generator import LCGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline

@component
class Summarizer:
    def __init__(self,model_name="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        self.model_name=model_name
        self.create_pipeline()

        self.summarized_documents_list=[]
    
    def create_pipeline(self):
        self.pipeline=Pipeline()

        prompt_template="""
            You are a personal assistant.
            Summarize the following context in a few words.
            Context:
            {{document.content}}
            Keep the relevant informations to answer the following question.
            Question:
            {{query}}
            Summary:
        """
        self.prompt_builder=PromptBuilder(template=prompt_template)

        self.generator=LCGenerator(model_name=self.model_name)

        self.pipeline.add_component(name="prompt_builder",instance=self.prompt_builder)
        self.pipeline.add_component(name="generator",instance=self.generator)

        self.pipeline.connect(sender="prompt_builder.prompt",receiver="generator.context")

    @component.output_types(summarized_documents=List[Document])
    def run(self,query:str,documents:List[Document]):
        self.summarized_documents_list=[]
        for document in documents:
            res=self.pipeline.run({"prompt_builder":{"query":query,"document":document}})
            self.summarized_documents_list.append(Document(content=res["generator"]["answer"],meta=document.meta))
        return {"summarized_documents":self.summarized_documents_list}

if __name__=="__main__":
    pipeline=Pipeline()

    summarizer=Summarizer()

    pipeline.add_component(name="summarizer",instance=summarizer)

    result=pipeline.run(data={"summarizer":{"query":"What is the capital of France ?","documents":[
        Document(content="La capitale de la France est Paris."),
        Document(content="Je ne suis pas certain de comprendre où cela va nous mener mais je sui sûr que ce n'est pas à Marseille, la plus grande ville du sud de la France."),
        Document(content="Les trois petits cochons viennent de Berlin. Cette ville n'est pas sans rappeler Londres avec ces grattes-ciel et ses larges rues."),
    ]}})
    print(result)