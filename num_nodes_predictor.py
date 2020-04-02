import tensorflow as tf
from dg_dagrnn import BroadHistoryRNN
import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse


import tensorflow as tf
import numpy as np
import collections
#import tensorlayer as tl
#from tensorlayer import logging
#from tensorlayer.decorators import deprecated_alias
#from tensorlayer.layers.core import Layer


NestedInput = collections.namedtuple('NestedInput', ['feature1', 'feature2'])


class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell
  
  @tf.function
  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.get_initial_state(input_data[0,:,:])
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state

class BroadHistoryCell(tf.keras.layers.Layer):
    def __init__(
        self,
        cell,
        name=None,  # 'rnn'
    ):

        self.cell = cell
        self.state_size = self.cell.units
        #self.output_size = self.cell
        super(BroadHistoryCell, self).__init__(name=name)

    def __repr__(self):
        s = ('{classname}(cell={cellname}, n_units={n_units}')
        s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, cellname=self.cell.__class__.__name__, n_units=self.cell.units,
            **self.__dict__
        )

    def build(self, input_shapes):
        # expect input_shape to contain 2 items, [(batch, i1), (batch, i2)]
        batch_size = input_shapes.feature1[0]
        num_nodes  = input_1_size = input_shapes.feature1[1]
        input_1_size = input_shapes.feature1[1]
        #print(f"b_size{batch_size}, max_nodes{num_nodes}, cell_dim{self.cell.units}")
        self.cell.build(input_shapes.feature1)
        self.previous_states = list()
        self.current_node_id = 0
        self.built = True

    def call(self, inputs, states):
        # inputs should be in [(batch, input_1), (batch, input_2)]
        # states: (batch_size, cell_dim)       
        #print(inputs)
        input_1 = inputs.feature1
        input_2 = inputs.feature2

        previous = states
        if len(self.previous_states) == 0:
        	self.previous_states.append(tf.zeros_like(previous))
        self.previous_states.append(states[0])
        with tf.init_scope():
        	history = tf.keras.layers.add(self.previous_states)
        output = self.cell(input_1, previous)
        self.current_node_id += 1
        return output

"""
class CustomRNN(tf.keras.layers.Layer):
    def __init__(
        self,
        cell,
        name=None,  # 'rnn'
    ):

        self.cell = cell
        self.state_size = self.cell.units
        #self.output_size = self.cell
        super(BroadHistoryCell, self).__init__(name=name)

    def __repr__(self):
        s = ('{classname}(cell={cellname}, n_units={n_units}')
        s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, cellname=self.cell.__class__.__name__, n_units=self.cell.units,
            **self.__dict__
        )

    def build(self, input_shapes):
        # expect input_shape to contain 2 items, [(batch,num_steps, i1), (batch,num_steps, i2)]
        batch_size = input_shapes.feature1[0]
        num_nodes  = input_1_size = input_shapes.feature1[1]
        input_1_size = input_shapes.feature1[1]
        #print(f"b_size{batch_size}, max_nodes{num_nodes}, cell_dim{self.cell.units}")
        self.cell.build(input_shapes.feature1)
        self.current_node_id = 0
        self.built = True

    def call(self, inputs):
        # inputs should be in [(batch,num_nodes,), (batch, input_2)]
        # states: (batch_size, cell_dim)       
        #print(inputs)
        input_1 = inputs.feature1
        input_2 = inputs.feature2

        for step in range(inputs.fea)
        output = self.cell(input_1, previous)
        

        self.current_node_id += 1
        return output
"""
class Network:
    def __init__(self):
        self._optimizer = tf.optimizers.Adam()
        #self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        #rnn = tf.keras.layers.RNN(BroadHistoryCell(tf.keras.layers.GRUCell(32)))
        #input_A = tf.keras.layers.Input(shape=(None, 8))
        input_X = tf.keras.layers.Input(shape=(8,3),batch_size=1)
        print("INPUTX",input_X.shape)
        hidden = DynamicRNN(tf.keras.layers.LSTMCell(32))(input_X)
        dense = tf.keras.layers.Dense(100, activation="relu")(hidden[0][:,-1,:])
        out = tf.keras.layers.Dense(1,activation="relu")(dense)
        self._optimizer = tf.optimizers.Adam()
        self.model = tf.keras.models.Model([input_X], out)
        self.model.summary()

    def train_batch(self, batch):
        with tf.GradientTape() as tape:
            print(batch.shape)
            estimate = self.model(batch[0],training=True)
            loss = tf.losses.mean_squared_error(batch[1], estimate)
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
            return loss
    def predict(self, x):
        return self.model(x, training=False)

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)


    X = []
    A = []
    Y = []

    i  = 0
    for sample in data:
        #print(sparse.csr.csr_matrix.todense(sample[0]),sparse.csr.csr_matrix.todense(sample[1]))
        G = nx.from_numpy_array(sparse.csr.csr_matrix.todense(sample[1]), create_using=nx.DiGraph)
        #print(f"Graph:{i}")
        a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
        x = sparse.csr.csr_matrix.todense(sample[0])
        #print(a)
        ordering = [x for x in nx.topological_sort(G)]
        #print("ORDER:",ordering)
        #print("reordered")
        X.append(x[ordering,:])
        A.append(a[ordering,:])
        Y.append(x.shape[0])
        #print(a[ordering,:])
        #print(x[ordering,:])
        #print(x.shape[0])
        #print("------------------------------------------")

    network = Network()

    print("Build complete")
    """
    #network.fit([X,A], np.array(Y),batch_size=1,epochs=100)
    print("prediction")
    print(Y)
    #print(network.predict([X[:10],A[:10]]))    

    batch_size = 20
    for epoch in range(20):
        indices = [np.random.randint(0,len(X)) for k in range(batch_size)]
        loss = 0
        for i in indices:
            batch = [np.expand_dims(X[i],axis=0),Y[i]]
            loss_b = network.train_batch(batch)
            #print(loss_b)
            loss += loss_b
        print(f"Epoch {epoch} - loss:{loss}")
    for i in range(10):
        print(network.predict((np.expand_dims(X[i],axis=0),np.expand_dims(A[i],axis=0))))           
    """