import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse
from dynamic_rnn import DynamicRNNEncoder

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
        self._optimizer = tf.optimizers.Adam()
        self.model = tf.keras.models.Model([input_X,input_A], out)
        self._metrics = {'loss': tf.keras.metrics.Poisson(), 'mse': tf.keras.metrics.MeanSquaredError()}
        #self.model.compile(loss=tf.keras.losses.poisson, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', 'mse'])
        self.model.summary()

    def train_batch(self, batch):
        #print(batch[0][0].shape)
        with tf.GradientTape() as tape:
            estimate = self.model(batch[0],training=True)
            loss = tf.keras.losses.poisson(batch[1], estimate)
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
            return loss
    
    def evaluate_batch(self, inputs, tags):
        new_tags = self.fix_tags(tags)
        tags_ex = np.expand_dims(new_tags, axis=2)
        mask = tf.not_equal(tags_ex, 0)
        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags_ex, probabilities, mask)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric(tags_ex, probabilities, mask)
    
    def evaluate(self, dataset, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch(
                [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                batch[dataset.TAGS].word_ids)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        return metrics

    def train(self, train_data, dev_data, args):
        for epoch in range(0,args.epochs):
            # batch_count = 0
            for batch in train_data.batches(args.batch_size):
                self.train_batch([batch[train_data.FORMS].word_ids,
                                  batch[train_data.FORMS].charseq_ids,
                                  batch[train_data.FORMS].charseqs],
                                 batch[train_data.TAGS].word_ids)
                # batch_count += 1
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, args)
            print("Dev accuracy: ", metrics['accuracy'])

if __name__ == "__main__":

    dataset = DAGDataset()
    dataset.read_nx_pickle(sys.argv[1])
    network = Network()

    print("Build complete")
    #network.model.fit([X,A], np.array(Y),batch_size=50,epochs=200,validation_split=0.2)
    #print(network.predict([X[:10],A[:10]]))    
    
    batch_size = 20
    epochs = 100
    for epoch in range(epochs):
        loss_b = 0
        for batch in dataset.batches(batch_size):
            loss_b += network.train_batch(batch)
        print(f"Epoch {epoch} - loss:{loss_b}")