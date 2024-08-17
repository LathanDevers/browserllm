from haystack import component
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack import Pipeline


@component
class QueryExtander:
    """
    A component expanding the query
    """
    def __init__(self):
        self.generator=LlamaCppGenerator(model="mistral-7b-instruct-v0.2.Q4_K_M.gguf",n_ctx=32768,n_batch=128, model_kwargs={"n_gpu_layers": -1},generation_kwargs={"temperature": 0.1})
        self.generator.warm_up()

    @component.output_types(extanded_query=str)
    def run(self,query:str):
        self.instruct = f"""You are a personal assistant.
        For each keyword of the question, add synonyms at the end of the question.
        Question: {query}
        Extanded question: """
        output = self.generator.run(prompt=f"[INST] {self.instruct} [/INST]",generation_kwargs={"max_tokens": 1024})
        response=output["replies"][0]
        return {"extanded_query":response}

if __name__=="__main__":
    pipeline=Pipeline()
    pipeline.add_component(instance=QueryExtander(),name="query_extander")
    results=pipeline.run({"query_extander":{"query":"Quelle est l'objectif principal de la fondation 21st Century Tiger?"}})
    print(results["query_extander"]["extanded_query"])


import yaml
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open("/content/question_answers.yml","r",encoding="utf-8") as f:
    questions_dict=yaml.safe_load(f)

qa_dict={}
c=0
for f in questions_dict.keys():
    for i in range(5):
        c+=1
        j=i*2
        q=questions_dict[f][j][f"question {i+1}"]
        try:
            a=questions_dict[f][j+1][f"réponse {i+1}"]
        except:
            a=questions_dict[f][j+1][f"reponse {i+1}"]
        try:
            qa_dict[f].update({q:a})
        except:
            qa_dict.update({f:{q:a}})
print(f"Nombre de questions : {c}")
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased',device=0)

from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.wsd import lesk
import json
lang='fra'

q_dict=[k for k in qa_dict.keys()]
stopwordsfr=stopwords.words('french')

def new_question(question):
    final_question=question+" ("

    question=question.replace("'"," ").replace("?","")
    ss=[]
    for l in question.split():
        if l.lower() not in stopwordsfr and l[0] not in [a for a in 'AZERTYUIOPQSDFGHJKLMWXCVBN1234567890']:
            temp="\""+l.lower()+"\" est un synonyme de \"[MASK]\"."
            synonyms=unmasker(temp)
            ss+=[s["token_str"] for s in synonyms]
    final_question+=", ".join(ss)
    final_question+=")"
    return final_question

from tqdm import tqdm
expanded_queries={}
for fichier in tqdm(qa_dict.keys(),desc="expand queries"):
    for question in tqdm(qa_dict[fichier],leave=False):
        expanded_query=new_question(question)
        expanded_queries.update({question:expanded_query})

with open("src/expanded_queries.json","w",encoding="utf-8") as eqf:
    json.dump(expanded_queries,eqf,ensure_ascii=False)