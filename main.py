from pipeline.rag_pipeline import RagPipeline
if __name__=="__main__":
    pipeline=RagPipeline(post_retrieval_options=["summary"])
    result=pipeline.run("Quelles sont les caractéristiques des chats ?")
    print(result)