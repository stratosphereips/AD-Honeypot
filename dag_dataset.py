#!/usr/bin/env python3
#Author: Ondrej Lukas - lukasond@fel.cvut.cz
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

class DAG_Dataset():
    def __init__(self, shuffle_batches=False, seed=42, max_samples=None):
        self._size = 0
        self._data = {"X":[], "A":[],"full_A":[], "full_X":[], "new_X":[], "target_A":[]}
        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
        self.max_samples = max_samples
    
    def read_nx_pickle(self, filename):
        with open(filename, "rb") as f:
            in_data = pickle.load(f)
        if self.max_samples:
            in_data = in_data[:self.max_samples]
        for sample in in_data:
            #transpose the matrix to get the orientation lower trianguler matrix
            a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
            #use 0 as padding
            x = sample[0] + 1
            self._data["X"].append(x)
            self._data["A"].append(a)
            self._size = len(self._data["X"])
    
    def size(self):
        return self._size
    
    @classmethod
    def loader(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    def batches(self, size):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]
                #find out maximum number of nodes in teh batch
                max_num_nodes_X = max(len(self._data["X"][i]) for i in batch_perm)
                #prepare array with dims[batchsize, maxnodes]
                batch_x = np.zeros([batch_size, max_num_nodes_X], np.float32)
                #prepare array with dims[batchsize, maxnodes, maxnodes]
                batch_a = -1*np.ones([batch_size, max_num_nodes_X, max_num_nodes_X], np.float32)

                for j in range(len(batch_perm)):
                    i = batch_perm[j]
                    #Node features
                    sample = self._data["X"][i]
                    #copy the values in the prepared 
                    batch_x[j,:len(sample)] = sample
                    
                    #Adjencency matrix
                    sample = self._data["A"][i]
                    batch_a[j,:sample.shape[0],:sample.shape[1]] = sample

                #create mask for adjacency matrix
                mask_a = batch_a != -1
                #set badding to 0 using the mask
                batch_a[~mask_a] = 0
                #print(mask_a)
                mask_a = np.tril(mask_a)
                yield {"X":batch_x,"A":batch_a,"mask_a":mask_a}