import json

with open("src/rewrited_queries.json",'r',encoding="utf-8") as f:
    d=json.load(f)
print(d.keys())