from flask import Flask,request,jsonify
from flask_cors import CORS
from pipeline.rag_pipeline import RagPipeline

app=Flask(__name__)
CORS(app)

@app.route('/api/data',methods=['POST'])
def receive_data():
    data=request.get_json()
    print(data)
    print(data["question"])
    pre_retrieval_option=data["preRetrievalOption"]
    retrieval_option=data["retrievalOption"]
    post_retrieval_option_d=data["postRetrievalOption"]
    post_retrieval_option=[]
    post_retrieval_type={}
    if post_retrieval_option_d["summary"]:
        post_retrieval_option.append("summary")
        post_retrieval_type.update({"summary":post_retrieval_option_d["summaryOption"]})
    if post_retrieval_option_d["rerank"]:
        post_retrieval_option.append("rerank")
        post_retrieval_type.update({"rerank":post_retrieval_option_d["rerankOption"]})
    if post_retrieval_option_d["fusion"]:
        post_retrieval_option.append("fusion")
        post_retrieval_type.update({"fusion":post_retrieval_option_d["fusionOption"]})


    generation_option=data["generationOption"]
    presentation_option=data["presentationOption"]
    question=data["question"]
    top_k=int(data["top_k"])

    question_pipeline=RagPipeline(pre_retrieval_options=pre_retrieval_option,retrieval_options=retrieval_option,post_retrieval_options=post_retrieval_option,post_retrieval_type=post_retrieval_type,generation_options=generation_option,index=index,top_k_retriever=top_k)
    results=question_pipeline.run(question)

    return jsonify({"status":"success","answer":results})

@app.route('/api/data',methods=['GET'])
def send_data():
    data={"message":"Depuis le serveur Flask"}
    return jsonify(data)

if __name__=='__main__':
    index="sentence"

    app.run(debug=True,port=5000,host='127.0.0.1')