import networkx as nx
#import plotly.graph_objects as go
from matplotlib import pylab
import random
import matplotlib.pyplot as plt
import itertools
from scipy import sparse
import numpy as np
import pickle
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib as mpl
#import pygraphviz
import math
#import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap('Set1')
import matplotlib.patches as mpatches
import sys
from sklearn.metrics import precision_recall_curve, roc_curve
from PIL import Image
import matplotlib.gridspec as gridspec


classes = {"padding":0,"DomainDNS":1, "Ou":2, "Container":3, "Group":4, "User":5, "New User":6}
classes_rev = {v:k for k,v in classes.items()}


def plot_graphs_comparison(g1,n1,g2,n2,g3,n3,legend,file_name, names):
    plt.axis('off')
    fig = plt.figure(figsize=(18, 8), dpi=100)
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=0.01, hspace=0.05)
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    ax3 = plt.subplot(gs1[2])
    ax1.title.set_text(names[0])
    ax2.title.set_text(names[1])
    ax3.title.set_text(names[2])
       #print("computing layout")
    pos =graphviz_layout(g1, prog='dot')

    nx.draw_networkx_nodes(g1,pos,node_color=n1,ax=ax1)
    nx.draw_networkx_edges(g1,pos,ax=ax1)
    nx.draw_networkx_labels(g1,pos,ax=ax1)

    pos2 =graphviz_layout(g2, prog='dot')

    nx.draw_networkx_nodes(g2,pos2,node_color=n2,ax=ax2)
    nx.draw_networkx_edges(g2,pos2,ax=ax2)
    nx.draw_networkx_labels(g2,pos2,ax=ax2)

    pos3 =graphviz_layout(g3, prog='dot')

    nx.draw_networkx_nodes(g3,pos3,node_color=n3,ax=ax3)
    nx.draw_networkx_edges(g3,pos3,ax=ax3)
    nx.draw_networkx_labels(g3,pos3,ax=ax3)

    fig.legend(handles=legend,loc='lower center',ncol=4)
    plt.savefig(file_name)
    pylab.close()
    del fig


def plot_graphs(g1,n1,g2,n2,legend,file_name, names):
    plt.axis('off')
    fig = plt.figure(figsize=(20,12))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text(names[0])
    ax2.title.set_text(names[1])
    pos =graphviz_layout(g1, prog='dot')

    nx.draw_networkx_nodes(g1,pos,node_color=n1,ax=ax1)
    nx.draw_networkx_edges(g1,pos,ax=ax1)
    nx.draw_networkx_labels(g1,pos,ax=ax1)

    pos2 =graphviz_layout(g2, prog='dot')

    nx.draw_networkx_nodes(g2,pos2,node_color=n2,ax=ax2)
    nx.draw_networkx_edges(g2,pos2,ax=ax2)
    nx.draw_networkx_labels(g2,pos2,ax=ax2)

    fig.legend(handles=legend,loc="lower center",ncol=4)
    plt.savefig(file_name)
    pylab.close()
    del fig

def save_graph(graph,file_name,node_colors, legend):
    plt.axis('off')
    fig = plt.figure()
    
    plt.subplot(121)
    pos =graphviz_layout(graph, prog='neato')

    nx.draw_networkx_nodes(graph,pos,node_color=node_colors)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)
    plt.subplot(122)
    pos =graphviz_layout(graph, prog='dot')

    nx.draw_networkx_nodes(graph,pos,node_color=node_colors)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    fig.legend(handles=legend,loc="lower center",ncol=4)
    plt.savefig(file_name)
    pylab.close()
    del fig

def plot_pr_curve(files, file_name):
    data = []
    for f in range(len(files)):
        with open(files[f], "rb") as infile:
            data.append(pickle.load(infile))
    fig, ax = plt.subplots(1, 1)
    colors = ["-r","-g","-b","-c"]
    labels = ["Dataset15", "Dataset50", "Dataset150", "Dataset500"]
    P = []
    R = []
    for f in range(len(files)):
        y_pred = []
        y_true = []
        for i in range(len(data[f]["A"])):
            real_size = np.argmin(data[f]["X"][i])
            if real_size == 0:
                real_size = len(data[f]["X"][i])
            y_true.append(np.reshape(data[f]["A"][i][:real_size,:real_size],[real_size*real_size]))
            y_pred.append(np.reshape(data[f]["A_hat"][i][:real_size,:real_size],[real_size*real_size]))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        P.append(precision)
        R.append(recall)

    for f in range(len(files)):
        ax.plot(R[f], P[f], colors[f], label=f"{labels[f]}")

    #plot random model results
    #find prior of the positive class
    unique, counts = np.unique(y_true, return_counts=True)
    prior = counts[1]/np.sum(counts)
    ax.plot([0, 1], [prior, prior], ':k',  label="Random")
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=3)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Model 3: Comparison of PR Curve per dataset")
    plt.savefig(file_name)

def plot_roc_curve(files, file_name):
    data = []
    labels = ["Dataset15", "Dataset50", "Dataset150", "Dataset500"]
    for f in range(len(files)):
        with open(files[f], "rb") as infile:
            data.append(pickle.load(infile))
    fig, ax = plt.subplots(1, 1)
    colors = ["-r","-g","-b","-c"]
    FPR = []
    TPR = []
    for f in range(len(files)):
        y_pred = []
        y_true = []
        for i in range(len(data[f]["A"])):
            real_size = np.argmin(data[f]["X"][i])
            if real_size == 0:
                real_size = len(data[f]["X"][i])
            y_true.append(np.reshape(data[f]["A"][i][:real_size,:real_size],[real_size*real_size]))
            y_pred.append(np.reshape(data[f]["A_hat"][i][:real_size,:real_size],[real_size*real_size]))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        FPR.append(fpr)
        TPR.append(tpr)
    for f in range(len(files)):
        ax.plot(FPR[f], TPR[f], colors[f], label=f"{labels[f]}")

    ax.plot([0, 1], [0, 1], ':k',  label="Random")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=3)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("Model 1: Comparison of ROC per dataset")
    plt.savefig(file_name)

def plot_Acomparison_rgb(files,threshold=0.3, loc="."):
   
    with open(files[0], "rb") as infile:
        data1 = pickle.load(infile)
    with open(files[1], "rb") as infile:
        data2 = pickle.load(infile)

    for i in range(len(data1["X"])):
        real_size = np.argmin(data1["X"][i])
        if real_size == 0:
            real_size = len(data1["X"][i])
        if np.sum(data1["A"][i] - data2["A"][i])!= 0:
            print("Pair of samples does not have the same ground truth")
        else:
            real = 255*(1-data1["A"][i]*data1["a_mask"][i][:real_size,:real_size])
            pred1= 255*(1-np.where(data1["A_hat"][i][:real_size,:real_size] >= threshold,1,0))
            pred2= 255*(1-np.where(data2["A_hat"][i][:real_size,:real_size] >= threshold,1,0))

            rgbA = np.zeros((real.shape[0], real.shape[1], 3),dtype=np.uint8)
            rgbA[...,0] = real
            rgbA[...,1] = pred1
            rgbA[...,2] = pred2
            im =  Image.fromarray(rgbA.astype(np.uint8))
            width, height = im.size
            newsize = (10*width, 10*height)
            im =  im.resize(newsize)  
            im.save(f"{loc}/{i}.png")


def plot_reconstructed_graphs(data, threshold, loc="."):
    for i in range(len(data["gen"])):
        x = data["X"][i]
        a = data["A"][i]
        a_hat = data["A_hat"][i]
        a_hat = np.where(a_hat > threshold,1,0)


        G_orig = nx.from_numpy_array(np.transpose(a), create_using=nx.DiGraph)
        G_pred = nx.from_numpy_array(np.transpose(a_hat), create_using=nx.DiGraph)

        legend = {}
        colors_orig = []
        colors_pred = []

        for j in range(len(x)):
            n = int(x[j])
            if n == 0:
                G_orig.remove_node(j)
            else:
                legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                colors_orig.append(cmap(n))
        for j in range(len(x)):
            n = int(x[j])
            if n == 0:
                G_pred.remove_node(j)
            else:
                legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                colors_pred.append(cmap(n))
        plot_graphs(G_orig,colors_orig, G_pred,colors_pred,legend.values(),f"./predictions/Type2/15/graph_{i}.png",["Original","Reconstructed"])

def plot_generated_graphs(data, loc="."):
    for i in range(len(data["gen"])):
        x = data["X"][i]
        a = data["A"][i]
        x_gen = data["gen"][i][0]
        a_gen = data["gen"][i][1]
    
        G_orig = nx.from_numpy_array(np.transpose(a), create_using=nx.DiGraph)
        G_pred = nx.from_numpy_array(np.transpose(a_gen), create_using=nx.DiGraph)

        legend = {}
        colors_orig = []
        colors_pred = []

        for j in range(len(x)):
            n = int(x[j])
            if n == 0:
                G_orig.remove_node(j)
            else:
                legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                colors_orig.append(cmap(n))
        for j in range(len(x_gen)):
            n = int(x_gen[j])
            if n == 0:
                G_pred.remove_node(j)
            elif j >= len(x):
                legend[6] = mpatches.Patch(color=cmap(0), label=classes_rev[6])
                colors_pred.append(cmap(0))
            else:
                legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                colors_pred.append(cmap(n))
        plot_graphs(G_orig,colors_orig, G_pred,colors_pred,legend.values(),f"./predictions/Type2/graph_{i}_generated.png",["Original",f"Extended ({len(x_gen)-len(x)} nodes)"])


def plot_generated_comparison(data,data2,loc="."):
    for i in range(len(data["gen"])):
        if np.sum(data["A"][i] - data2["A"][i]) != 0:
            print("Pair fo samples does not match!")
        else:
            x = data["X"][i]
            a = data["A"][i]
            x_gen = data["gen"][i][0]
            a_gen = data["gen"][i][1]
            a_gen2 = data2["gen"][i][1]
        
            G_orig = nx.from_numpy_array(np.transpose(a), create_using=nx.DiGraph)
            G_pred = nx.from_numpy_array(np.transpose(a_gen), create_using=nx.DiGraph)
            G_pred2 = nx.from_numpy_array(np.transpose(a_gen2), create_using=nx.DiGraph)
            legend = {}
            colors_orig = []
            colors_pred = []
            colors_pred2 = []

            for j in range(len(x)):
                n = int(x[j])
                if n == 0:
                    G_orig.remove_node(j)
                else:
                    legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                    colors_orig.append(cmap(n))
            
            for j in range(len(x_gen)):
                n = int(x_gen[j])
                if n == 0:
                    G_pred.remove_node(j)
                    G_pred2.remove_node(j)
                elif j >= len(x):
                    legend[6] = mpatches.Patch(color=cmap(0), label=classes_rev[6])
                    colors_pred.append(cmap(0))
                else:
                    legend[n] = mpatches.Patch(color=cmap(n), label=classes_rev[n])
                    colors_pred.append(cmap(n))
            plot_graphs_comparison(G_orig, colors_orig, G_pred, colors_pred, G_pred2, colors_pred,legend.values(),f"./predictions/generated_comparison1_2_15/{i}.png",names=["Original", "Model1", "Model2"])

def check_connectivity(data):
    unconnected = 0
    total_added = 0
    original_users = 0
    original_user_edges = 0
    generated_user_edges = 0
    for i in range(len(data["gen"])):
        x = data["X"][i]
        a = data["A"][i]
        x_gen = data["gen"][i][0]
        a_gen = data["gen"][i][1]
        #count connections
        connections = np.sum(a_gen,axis=1)
        #take only the new nodes
        unconnected += np.sum(np.where(connections[len(x):] == 0, 1,0))
        generated_user_edges += np.sum(connections[len(x):])
        total_added += len(x_gen) - len(x)
    
        unique, counts = np.unique(x, return_counts=True)
        original_users += counts[-1]
        original_user_edges += np.sum(np.sum(a,axis=1)[x == 5])
    
    return unconnected/total_added, (generated_user_edges/total_added)/(original_user_edges/original_users)

def validity_ratio(data):
    predicted_edges =  0
    valid_edges = 0
    for i in range(len(data["gen"])):
        x = data["X"][i]
        a = data["A"][i]
        x_gen = data["gen"][i][0]
        a_gen = data["gen"][i][1]
        G = nx.from_numpy_array(np.transpose(a_gen), create_using=nx.DiGraph)
        #count connections
        connections = np.sum(a_gen,axis=1)
        #take only the new nodes
        predicted_edges += np.sum(connections[len(x):])
        for n in range(len(x), len(x_gen)):
            in_ou = False
            for edge in G.in_edges(n):
                parent = edge[0]
                if x_gen[parent] == 2:
                    if not in_ou:
                        valid_edges += 1
                        in_ou = True
                elif x_gen[parent] == 3 or x_gen[parent] == 4:
                    valid_edges += 1
    return valid_edges/predicted_edges