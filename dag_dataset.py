import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse
import numpy as np
import collections
from datetime import datetime
import os
import re
import random


class DAG_edge_dataset():

    def __init__(self, shuffle_batches=False, seed=42, pred_percentage=0):
        self._size = 0
        self._data = {"X":[], "A":[],"full_A":[], "full_X":[], "new_X":[], "target_A":[]}
        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
        self._pred_percentage = pred_percentage

    def read_nx_pickle(self, filename):
        with open(filename, "rb") as f:
            in_data = pickle.load(f)
        for sample in in_data:
            a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
            x = sample[0] +1
            self._data["full_X"].append(x)
            self._data["full_A"].append(a)

            
            #reconstruct graph
            G = nx.from_numpy_array(sparse.csr.csr_matrix.todense(sample[1]), create_using=nx.DiGraph)
            #find leaf nodes
            leafs = [node for node in G.nodes() if G.out_degree(node) == 0]
            #randomly seledt one to remove
            to_remove = random.sample(leafs, 1)[-1]
            
            #construct list of edges for the node
            target_A = []
            new_X = x[to_remove]
            for n in G.nodes():
                if n == to_remove:
                    continue
                if G.has_edge(n,to_remove):
                    target_A.append(1)
                else:target_A.append(0)

            target_A = np.array(target_A)
            new_X = x[to_remove]
            X = []
            for i in range(len(x)):
                if i == to_remove:                   
                    continue
                X.append(x[i])

            G.remove_node(to_remove)
            A = nx.to_numpy_array(G).transpose()
            self._data["X"].append(X)
            self._data["A"].append(A)
            self._data["target_A"].append(target_A)
            self._data["new_X"].append(new_X)
            self._size = len(self._data["X"])
    
    def size(self):
        return self._size
    def store_to_pickle(self, filename):
        with open (filename, "wb") as f:
            pickle.dump(self,f)
    def load_from_pickle(self, filename):
    def batches(self, size):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                
                batch_full_A = []
                batch_full_X = []
    
                max_num_nodes_X = max(len(self._data["X"][i]) for i in batch_perm)
                max_num_nodes_new_X = 1
                
                batch_x = np.zeros([batch_size, max_num_nodes_X], np.float32)
                batch_new_x = np.zeros(batch_size, np.float32)
                batch_a = np.zeros([batch_size, max_num_nodes_X, max_num_nodes_X], np.float32)
                batch_target_a = np.zeros([batch_size, max_num_nodes_X], np.float32)
                for j in range(len(batch_perm)):
                    i = batch_perm[j]
                    #Node features
                    sample = self._data["X"][i]
                    batch_x[j,:len(sample)] = sample
                    new_x = self._data["new_X"][i]
                    batch_new_x[j] = new_x
                    
                    #Adjencency matrices
                    #A = np.pad(self._data["A"][i],((0,max_num_nodes_X-self._data["A"][i].shape[0]),(0,max_num_nodes_X-self._data["A"][i].shape[0])),'constant',constant_values=(0,0))
                    #batch_a.append(A)
                    sample = self._data["A"][i]
                    batch_a[j,:sample.shape[0],:sample.shape[1]] = sample

                    sample = self._data["target_A"][i]
                    batch_target_a[j,:sample.shape[0]] = sample
                    
                    batch_full_X.append(self._data["full_X"][i])
                    batch_full_A.append(self._data["full_A"][i])

                #print("Prepared batchx:")
                #print(batch_x)
                yield {"full_X": batch_full_X, "full_A":batch_full_A,
                "X":batch_x,"A":batch_a, "new_X": np.expand_dims(batch_new_x,axis=1), "target_A":batch_target_a}


