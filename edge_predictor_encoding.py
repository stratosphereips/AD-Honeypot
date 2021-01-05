#!/usr/bin/env python3
#Author: Ondrej Lukas - lukasond@fel.cvut.cz
import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse
from dynamic_rnn import DynamicRNNEncoder,DynamicRNNDecoder,EdgePredictor

import tensorflow as tf
import numpy as np
import collections
from datetime import datetime
import os
import re
import random


def frobenius_loss(X,Y):
    norm = tf.norm(X-Y,ord='fro',axis=[-2,-1])      
    #norm = tf.math.divide(norm,tf.cast(sizes,tf.float32))
    return tf.math.reduce_mean(norm)

def mse(X,Y):
    return tf.math.reduce_sum(tf.keras.losses.mean_absolute_error(Y,X))

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


"""
class Network:
    def __init__(self,args):
        self._feature_dim = args.feature_dim
        self._optimizer = tf.optimizers.Adam(learning_rate=args.lr)
        self._edge_threshold = args.edge_threshold

        #INPUTS
        input_A = tf.keras.layers.Input(shape=(None,None),name="Input_A")
        input_X = tf.keras.layers.Input(shape=(None,), name="Input_X")
        input_new_nodes = tf.keras.layers.Input(shape=(None,), name="Input_new_X")

        #EMBEDDING
        node_embedding = tf.keras.layers.Embedding(input_dim=8, output_dim=8)
        #ENCODER
        encoded1 = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, embed_layer=node_embedding, return_sequences=True)(input_X,input_A)
        backwards1 = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, embed_layer=node_embedding, return_sequences=True, go_backwards=True)(input_X,input_A)
        encoded1 = tf.keras.layers.Add()([encoded1,backwards1])
    
        encoded2 = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, embed_layer=None, return_sequences=True)(encoded1,input_A)
        backwards2 = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, embed_layer=None, return_sequences=True, go_backwards=True)(encoded1,input_A)
        encoded2 = tf.keras.layers.Add()([encoded2,backwards2])

        #predcitions
        predicted_A = EdgePredictor(emb_layer=node_embedding)(encoded1, input_new_nodes)
  
        self.model = tf.keras.Model([input_X, input_A, input_new_nodes], predicted_A,name="Edge_predictor")

        #losses and metrics
        self._loss = tf.keras.losses.BinaryCrossentropy()
        self._num_edge_loss = tf.keras.losses.MeanSquaredError()
        self._metrics = {"loss":tf.keras.metrics.Mean(), "accuracy":tf.keras.metrics.BinaryAccuracy(threshold=args.edge_threshold)}
        #logging
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
   
    #@tf.function
    def train_batch(self, batch):
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]
        target_A = batch["target_A"]
        
        with tf.GradientTape() as tape:
            #predict A
            predicted_A = tf.squeeze(self.model([X,A,new_X], training=True))
            
            #compute loss
            loss = self._loss(tf.expand_dims(target_A,axis=2),predicted_A)
            
            #make step
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            #gradient_norm = -1
            #gradients, gradient_norm = tf.clip_by_global_norm(gradients, 2)

            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)


            #compute accuracy:
            #binary = tf.where(predicted_A >= self._edge_threshold, 1,0)
            #binary = predicted_A
            #accuracy = tf.math.reduce_mean(1-tf.math.reduce_sum(tf.math.abs(binary-A),axis=(1,2))/(A.shape[1]*A.shape[0]))
            #log metrics
            with self._writer.as_default():
                tf.summary.scalar("training/loss", loss)
                #tf.summary.scalar("training/gradient_norm", gradient_norm)
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
                for var in self.model.trainable_variables:
                    tf.summary.histogram(var.name, var)
            return loss

    def train(self, train_data, dev_data, epochs, batch_size):
        for epoch in range(0,epochs):
            for batch in train_data.batches(batch_size):
                self.train_batch(batch)
            #Evaluate on dev data
            metrics = network.evaluate(dev_data, batch_size)
            tf.print(f"EPOCH {epoch+1}/{epochs}:\tDev loss:{metrics['loss']}, A_acc:{metrics['accuracy']}")
    
    

    def evaluate_batch(self, batch):
        #input preparation
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]
        target_A = batch["target_A"]

        predicted_A = self.model([X,A,new_X], training=False)
        #compute loss
        loss = self._loss(tf.expand_dims(target_A,axis=2),predicted_A)

        #logging
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            elif name == "accuracy":
                metric(tf.expand_dims(target_A,axis=2),predicted_A)
    
    def evaluate(self, dataset, batch_size):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(batch_size):
            self.evaluate_batch(batch)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("DAG_small_dev", name), value)
        return metrics

    def test(self,batch,threshold=0.5):
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]

        predicted_A = self.model([X,A,new_X], training=False)
        binary = tf.where(predicted_A >= threshold, 1,0)
        return predicted_A,binary
"""
class Network:

    class Model(tf.keras.Model):
        def __init__(self, args, node_class_dim):
            super().__init__()
            self.node_embedding = tf.keras.layers.Embedding(input_dim=node_class_dim, output_dim=args.node_emb_dim)
            self.encoder_forward = DynamicRNNEncoder(rnn_dim=args.rnn_dim, return_sequences=True)
            self.encoder_backwards = DynamicRNNEncoder(rnn_dim=args.rnn_dim, return_sequences=True,go_backwards=True)
            self.edge_predictor = EdgePredictor()
        """
        def call(self,inputs):
            #assert isinstance(inputs, list) and len(inputs) == 3
            node_inputs, edges, target_nodes = inputs

            in_embedding = self.node_embedding(node_inputs)
            next_in_embedding = self.node_embedding(node_inputs)
            next_in_embedding = self.node_embedding(node_inputs)        
            
            #get hidden states for nodes
            enc_forward = self.encoder_forward(in_embedding, edges)
            enc_backwards = self.encoder_backwards(in_embedding, edges)
            merged = tf.keras.layers.Add()([enc_forward, enc_backwards])
            predictions = self.edge_predictor(merged, next_in_embedding)
            return predictions
        """
    
    def __init__(self, args, node_class_dim):
        self.model = self.Model(args, node_class_dim)

        self._feature_dim = args.feature_dim
        self._optimizer = tf.optimizers.Adam(learning_rate=args.lr)
        self._edge_threshold = args.edge_threshold
        self._loss = tf.keras.losses.BinaryCrossentropy()
        self._num_edge_loss = tf.keras.losses.MeanSquaredError()
        self._metrics = {"loss":tf.keras.metrics.Mean(), "accuracy":tf.keras.metrics.BinaryAccuracy(threshold=args.edge_threshold)}
        #logging
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    #@tf.function
    def train_batch(self, batch):
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]
        target_A = batch["target_A"]
        
        with tf.GradientTape() as tape:
            """
            #predict A
            predicted_A = tf.squeeze(self.model([X,A,new_X], training=True))
            
            #compute loss
            loss = self._loss(tf.expand_dims(target_A,axis=2),predicted_A)
            
            #make step
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            #gradient_norm = -1
            #gradients, gradient_norm = tf.clip_by_global_norm(gradients, 2)

            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)


            #compute accuracy:
            #binary = tf.where(predicted_A >= self._edge_threshold, 1,0)
            #binary = predicted_A
            #accuracy = tf.math.reduce_mean(1-tf.math.reduce_sum(tf.math.abs(binary-A),axis=(1,2))/(A.shape[1]*A.shape[0]))
            #log metrics
            with self._writer.as_default():
                tf.summary.scalar("training/loss", loss)
                #tf.summary.scalar("training/gradient_norm", gradient_norm)
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
                for var in self.model.trainable_variables:
                    tf.summary.histogram(var.name, var)

            return loss
            """
        with tf.GradientTape() as tape:
            in_embedding = self.model.node_embedding(X)
            forward = self.model.encoder_forward(in_embedding,A)
            backwards self.model.enc_backwards(in_embedding,A)
            merged = tf.keras.layers.Add([forward,backwards])
            next_in_embedding = self.model.node_embedding(new_X)
            predcitions = tf.squeeze(self.model.edge_predictor(merged,next_X))


            loss = self._loss(tf.expand_dims(target_A,axis=2),predcitions)

    def train(self, train_data, dev_data, epochs, batch_size):
        for epoch in range(0,epochs):
            for batch in train_data.batches(batch_size):
                self.train_batch(batch)
            #Evaluate on dev data
            metrics = network.evaluate(dev_data, batch_size)
            tf.print(f"EPOCH {epoch+1}/{epochs}:\tDev loss:{metrics['loss']}, A_acc:{metrics['accuracy']}")
    
    

    def evaluate_batch(self, batch):
        #input preparation
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]
        target_A = batch["target_A"]

        predicted_A = self.model([X,A,new_X], training=False)
        #compute loss
        loss = self._loss(tf.expand_dims(target_A,axis=2),predicted_A)

        #logging
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            elif name == "accuracy":
                metric(tf.expand_dims(target_A,axis=2),predicted_A)
    
    def evaluate(self, dataset, batch_size):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(batch_size):
            self.evaluate_batch(batch)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("DAG_small_dev", name), value)
        return metrics

    def test(self,batch,threshold=0.5):
        X = batch["X"]
        A = batch["A"]
        new_X = batch["new_X"]

        predicted_A = self.model([X,A,new_X], training=False)
        binary = tf.where(predicted_A >= threshold, 1,0)
        return predicted_A,binary

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    #global parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    #encoder parameters
    parser.add_argument("--feature_dim", default=8, type=int, help="Feature dimension")
    parser.add_argument("--rnn_dim", default=64, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--node_emb_dim", default=4, type=int, help="Graph embedding dimension")
    parser.add_argument("--edge_threshold", default=0.3, type=float, help="Edge limit value during generation")
    args = parser.parse_args()
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    #set amount of cores to use
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)



    #train_data = "./dag-data/DAG_small_generated/train/train_small.pickle"
    #dev_data = "./dag-data/DAG_small_generated/dev/dev_small_dag.pickle"
    train_data = "./dag-data/DAG_same_size_generated/train/train_size_8.pickle"
    dev_data = "./dag-data/DAG_same_size_generated/dev/dev_size_8.pickle"
    #train_data = "./dag-data/DAG_8_15_generated/train/train_small8_15.pickle"
    #dev_data = "./dag-data/DAG_8_15_generated/dev/dev_small8_15.pickle"
    #dev_data = "./dag-data/dummy_dataset/train/train_dummy.pickle"

    dataset_train = DAG_edge_dataset(shuffle_batches=True,seed=42)
    
    dataset_train.read_nx_pickle(train_data)
    dataset_dev = DAG_edge_dataset(shuffle_batches=True,seed=42)
    dataset_dev.read_nx_pickle(dev_data)
    dataset_test = DAG_edge_dataset()
    dataset_test.read_nx_pickle(dev_data)
    network = Network(args,8)

    tf.print("Build complete")
    tf.print("TRAIN size:", dataset_train.size())
    tf.print("DEV size:", dataset_dev.size())
    

    network.train(dataset_train, args)
    
    """
    count = 5
    data = dataset_dev.batches(count)
    batch = next(data)
    raw,predicted_A = network.test(batch,args.edge_threshold)
    predictions = []
    for i in range(count):
        gA = batch["target_A"][i]
        #gF = batch["full_X"][i]
        #pA = predicted_A[i,:len(gF),:len(gF)]
        #pF = gF
        tf.print("predicted:",output_stream="file://output.txt")
        tf.print(tf.cast(predicted_A[i,:,:],tf.float32).numpy(),output_stream="file://output.txt")
        tf.print("predicted_raw:",output_stream="file://output.txt")
        tf.print(tf.cast(raw[i,:,:],tf.float32).numpy(),output_stream="file://output.txt")
        tf.print("target:",output_stream="file://output.txt")
        tf.print(tf.cast(batch["target_A"][i],tf.int32).numpy(),output_stream="file://output.txt")
        tf.print("---------------------------------------",output_stream="file://output.txt")
        #predictions.append((gF,gA,pF,pA))
    #with open("predicted.pickle", "wb") as outfile:
    #    pickle.dump(predictions, outfile)
    """