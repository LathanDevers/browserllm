import matplotlib.pyplot as plt

def plot_i(title:str,tp_list):

    plt.plot(range(0,300,10),tp_list[1],color='r',label="embeddings retriever")
    plt.plot(range(0,300,10),tp_list[2],color='g',label="bm25 retriever")
    plt.plot(range(0,300,10),tp_list[3],color='b',label="rewrite + embeddings retriever")
    plt.plot(range(0,300,10),tp_list[4],color='m',label="expand + embeddings retriever")
    plt.plot(range(0,300,10),tp_list[5],color='c',label="flare + embeddings retriever")
    plt.plot(range(0,300,10),tp_list[6],color='k',label="HyDE + embeddings retriever")

    plt.xlabel("nombre de documents")
    plt.ylabel("nombre de bon documents")
    plt.title(title)
    plt.legend()

    plt.show()