import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse
from dynamic_rnn import DynamicRNNEncoder,DynamicRNNDecoder

import tensorflow as tf
import numpy as np
import collections
import recurrent_fixed


class DAGDataset():

    def __init__(self,shuffle_batches=False):
        self._size = 0
        self._data = {"features":[], "conditions":[]}
        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    def read_nx_pickle(self, filename):
        with open(sys.argv[1], "rb") as f:
            in_data = pickle.load(f)
        for sample in in_data:
            G = nx.from_numpy_array(sparse.csr.csr_matrix.todense(sample[1]), create_using=nx.DiGraph)
            a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
            x = sparse.csr.csr_matrix.todense(sample[0])
            ordering = [k for k in nx.topological_sort(G)]
            self._data["features"].append(x[ordering,:])
            self._data["conditions"].append(a[ordering,:])
        self._size = len(self._data["features"])

    def size(self):
        return self._size

    def batches(self, size):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch_x = []
                batch_a = []
                batch_targets = []
                lengths = [self._data["features"][i].shape[0] for i in batch_perm]
                #print(lengths)
                max_num_nodes = max(self._data["features"][i].shape[0] for i in batch_perm)
                for i in batch_perm:
                    #Node features
                    if self._data["features"][i].shape[0] < max_num_nodes:
                        #Node features
                        X = np.pad(self._data["features"][i],((0,max_num_nodes-self._data["features"][i].shape[0]),(0,0)),'constant',constant_values=(0,0))
                        #Adjencency matrices
                        A = np.pad(self._data["conditions"][i],((0,max_num_nodes-self._data["conditions"][i].shape[0]),(0,max_num_nodes-self._data["conditions"][i].shape[0])),'constant',constant_values=(0,0))
                    #Adjencency matrices
                    #Targets
                    else:
                        X = self._data["features"][i]
                        A = self._data["conditions"][i]
                    target = self._data["features"][i].shape[0]
                    batch_x.append(X)
                    batch_a.append(A)
                    batch_targets.append(target)
                yield (np.array(batch_x,dtype=np.float32), np.array(batch_a,dtype=np.float32)), np.array(batch_targets,dtype=np.float32)



class Network:
    def __init__(self):
        self._optimizer = tf.optimizers.Adam()
        #self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        input_A = tf.keras.layers.Input(shape=(None,None,))
        input_X = tf.keras.layers.Input(shape=(None,3,))
        rnn = DynamicRNNEncoder(recurrent_fixed.GRUCell(32),tf.keras.layers.Dense(32, activation="relu"),tf.keras.layers.Dense(32, activation="relu"))(input_X,input_A)
        dense = tf.keras.layers.Dense(100, activation="relu")(rnn)
        out = tf.keras.layers.Dense(1,activation="relu")(dense)

        F,A,max_node = DynamicRNNDecoder(recurrent_fixed.GRUCell(16),32,3)(rnn,out)

        self._optimizer = tf.optimizers.Adam(clipnorm=True)
        self.model = tf.keras.models.Model([input_X,input_A], [out,F,A,max_node])
        self._metrics = {'loss': tf.metrics.Mean(), 'mse': tf.keras.metrics.MeanSquaredError()}
        self._loss_fn = tf.keras.losses.Poisson()
        #self.model.compile(loss=tf.keras.losses.poisson, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', 'mse'])
        self.model.summary()

    def combined_loss(self,golden, model_output):
        means, F_hat,A_hat,max_node = model_output
        num_nodes, F, A = golden
        tf.print("Max_node:",max_node, max_node.shape)
        #tf.print(means, np.expand_dims(num_nodes,axis=1),tf.keras.losses.poisson(means,np.expand_dims(num_nodes,axis=1)))
        #padd targets so we can compute loss
        F_ex = []
        A_ex = []
        for i in range(F.shape[0]):
            tf.print(f"predicted_nodes:{max_node}, target_N:{np.max(num_nodes)}")
            F_ex.append(np.pad(F[i,:,:],((0,max(max_node,F.shape[1])-F.shape[1]),(0,max(max_node,F.shape[2])-F.shape[2])),'constant',constant_values=0))
            A_ex.append(np.pad(A[i,:,:],((0,max(max_node,A.shape[1])-A.shape[1]),(0,max(max_node,F.shape[2])-F.shape[2])),'constant',constant_values=0))
        tf.print(F_ex[0].shape, A_ex[0].shape, F_hat.shape, A_hat.shape)
        #tf.print(f"Mean:{means.shape},F_hat:{F_hat.shape},A_hat:{A_hat.shape}")
        #tf.print(f"F:{F.shape},A:{A.shape}")
        #if max_node == 1:
        #    tf.print("SHIT!")
        #    return tf.keras.losses.poisson(np.expand_dims(num_nodes,axis=1),means)
        #else:
        loss = tf.keras.losses.poisson(np.expand_dims(num_nodes,axis=1),means) + self.structural_loss(A_ex,A_hat) #+ self.feature_loss(F_ex,F_hat)
        #tf.print("Combined:", tf.metrics.Mean(loss))
        return loss
    
    def structural_loss(self,A,A_hat):
        return tf.norm(A-A_hat)
  
    def feature_loss(self, F, F_hat):
        return tf.norm(F-F_hat)
    
    def train_batch(self, batch):
        #print(batch[0][0].shape)
        with tf.GradientTape() as tape:
            estimate = self.model(batch[0],training=True)
             
            #loss = self._loss_fn(np.expand_dims(batch[1],axis=1), estimate)
            loss = self.combined_loss([batch[1],batch[0][0], batch[0][1]],estimate)
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
            return loss
    
    def evaluate_batch(self, inputs, targets):
        #tf.print(targetgets.shape)
        targets_ex = np.expand_dims(targets,axis=1)
        means,n_features,A,max_node = self.model(inputs, training=False)
        
        #tf.print(means.shape, targets_ex.shape)
        loss = self._loss_fn(targets_ex,means)
        self.combined_loss([targets,inputs[0], inputs[1]], [means,n_features,A], max_node)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric(means, targets)
    
    def evaluate(self, dataset, batch_size):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(batch_size):
            self.evaluate_batch(batch[0], batch[1])

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        return metrics

    def train(self, train_data, dev_data, epochs, batch_size):
        for epoch in range(0,epochs):
            # batch_count = 0
            for batch in train_data.batches(batch_size):
                self.train_batch(batch)
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, batch_size)
            print(f"EPOCH {epoch}/{epochs}:\tDev loss:{metrics['loss']}, Dev MSE:{metrics['mse']}")

if __name__ == "__main__":

    dataset_train = DAGDataset()
    dataset_train.read_nx_pickle(sys.argv[1])
    dataset_dev = DAGDataset()
    dataset_dev.read_nx_pickle(sys.argv[2])
    network = Network()

    print("Build complete")
    #network.model.fit([X,A], np.array(Y),batch_size=50,epochs=200,validation_split=0.2)
    #print(network.predict([X[:10],A[:10]]))    
    
    batch_size = 4
    epochs = 1
    network.train(dataset_train, dataset_dev, epochs,batch_size)