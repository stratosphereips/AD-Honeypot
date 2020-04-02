import networkx as nx
#import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
import itertools
from scipy import sparse
import numpy as np
import pickle
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib as mpl
import pygraphviz
import math
#import matplotlib.pyplot as plt


def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    root = 0
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = None
        for k in range(len(G.degree)):
            if G.degree[k] == 0:
                a = k
                break
        if not a:
            a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G) and nx.has_path(G,b,root):
            edges -= 1
        else:
            #closed loop or another component in graph
            G.remove_edge(a,b)
    G = G.reverse()
    return G

def get_random_color():
    return tuple([random.random() for k in range(3)])



graphs = []
for i in range(0,100):
    n_nodes = 8
    G = random_dag(nodes=n_nodes,edges=random.randint(n_nodes-1,math.floor(n_nodes+n_nodes/3)))
    degrees = [G.degree[x] for x in G.nodes]
    n_colors =  [get_random_color() for k in degrees]

    X = np.array([list(c) for c in n_colors])
    #print(n_colors)
    #print(sparse.csr.csr_matrix(X))
    #nx.draw_planar(G,node_color=n_colors,with_labels=True)
    #pos=graphviz_layout(G, prog='dot')
    #G.reverse()
    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos,arrows=True,node_color=n_colors,with_labels=True)
    plt.savefig(f"graph{i}.png")
    plt.clf()
    #print(sparse.csr_matrix(nx.to_numpy_array(G)))
    graphs.append((sparse.csr.csr_matrix(X),sparse.csr_matrix(nx.to_numpy_array(G))))
with open('test_graph_dataset.pickle', 'wb') as handle:
    pickle.dump(graphs, handle)