from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from spektral.layers import GraphConv
import tensorflow as tf
import pickle
import sys
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib as mpl
import pygraphviz
import spektral
class Network:
    def __init__(self, N,F, lr=1e-3,gamma=0.5):

        self.gamma = gamma
        l2_reg = 5e-2 / 2
        #ENCODER
        X_in = Input(shape=(F, ))
        A_in = Input((N, ), sparse=False)

        hidden = spektral.layers.GraphAttention(64, attn_heads=5, concat_heads=True, dropout_rate=0.2, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([X_in, A_in])
        hidden = spektral.layers.GraphAttention(128, attn_heads=5, concat_heads=True, dropout_rate=0.2, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([hidden, A_in])
        hidden = spektral.layers.GraphAttention(1, attn_heads=5, concat_heads=False, dropout_rate=0.2, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([hidden, A_in])
        A_pred = A_hat = spektral.layers.InnerProduct(trainable_kernel=True, activation="relu", kernel_initializer='glorot_uniform')(hidden)
        self.encoder = tf.keras.Model(inputs=[X_in, A_in], outputs=[A_pred])

        #X2_in = Input(shape=(1, ))
        #approximate Adjacency matrix
        #A_hat = spektral.layers.InnerProduct(trainable_kernel=True, activation="relu", kernel_initializer='glorot_uniform')(X2_in)

        #approximate feature matrix
        #hidden = spektral.layers.GraphAttention(64, attn_heads=5, concat_heads=True, dropout_rate=0.2, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([X2_in, A_hat])
        #hidden = spektral.layers.GraphAttention(128, attn_heads=5, concat_heads=True, dropout_rate=0.2, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([hidden, A_hat])
        #X_hat = spektral.layers.GraphAttention(F, attn_heads=5, concat_heads=False, dropout_rate=0.5, return_attn_coef=False, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform')([hidden, A_hat])
        
        #self.decoder = tf.keras.Model(inputs=[X2_in], outputs=[X_hat, A_hat])

        self._optimizer = tf.optimizers.Adam()
        self._reconstruction_loss_fn = tf.losses.BinaryCrossentropy()

    def _kl_divergence(self, a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / b_sd_squared
        return (a_mean - b_mean) ** 2 / (2 * b_sd_squared) + (ratio - tf.math.log(ratio) - 1) / 2

    
    def _loss_fn(self, X,X_hat, A,A_hat):

        #print(X.shape, X_hat.shape, A.shape, A_hat.shape)




        l = (tf.norm((X-X_hat),ord='fro',axis=(0,1))**2) + (tf.norm((A-A_hat),ord='fro',axis=(0,1))**2)
        return l
   

    #@tf.function
    def train_batch(self, batch):
        X,A = batch
        #print(type(A), "typeA")
        fltr = GraphConv.preprocess(A).astype('f4')
        X = X.toarray()
        #print(fltr)
        #print("TYPE:",type(fltr))
        #print(X)
        fltr = fltr.toarray()
        #print(fltr.shape.ndim)

        #print("SHAPES:",X.shape, fltr.shape)
        with tf.GradientTape() as tape:
            #encode graph and get the matrix

            H = self.encoder([X,fltr])
            #Xgen, A_hat = self.decoder([H])


            #loss = self._loss_fn(X,Xgen, fltr,A_hat)
            loss = (tf.norm((fltr-H),ord='fro',axis=(0,1))**2)

            variables = self.encoder.trainable_variables# + self.decoder.trainable_variables
            gradients = tape.gradient(loss,variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
            return loss


    def train_epoch(self, batches, args):
        loss = 0
        for batch in batches:
            #print(batch)
            loss += self.train_batch(batch)
        #self.generate()
        return loss


with open(sys.argv[1], 'rb') as handle:
    graphs_in = pickle.load(handle)
X = graphs_in[0][0]
graphs = graphs_in
N = X.shape[0]  # Number of nodes in the graph
F = X.shape[1]
epochs = 20

print("********************************")
print(graphs[0][0].shape)


print(f"N:{N}, F:{F}")
network = Network(N,F)


for epoch in range(epochs):
    loss = 0
    for batch in graphs:
        #print(batch)
        loss += network.train_batch(list(batch))
    print(f"Epoch {epoch} - Loss={loss}")


k = 0
for batch in graphs_in[:3]:

    X,A = batch
    fltr = GraphConv.preprocess(A).astype('f4')
    X = X.toarray()

    #print("TYPE:",type(fltr))
    #print(X)
    fltr = fltr.toarray()
    H = network.encoder([X,fltr])
    Xgen, A_hat = network.decoder(H)

    np_matrix = np.asmatrix(A_hat.numpy())
    np_matrix2 = np_matrix
    print(np_matrix2)
    #print("RES:",np_matrix.shape)
    #np_matrix2 = np_matrix -2*np.tril(np_matrix)
    #np.fill_diagonal(np_matrix2, 0)
    #np_matrix2[np_matrix2 < 0.] = 0
    #print(np_matrix2)
    #np_matrix2 = np_matrix2.astype(int)
    #print("###########################")
    G=nx.from_numpy_matrix(np_matrix2, create_using=nx.DiGraph())
    Xgen = Xgen.numpy().astype(float)
    n_colors = []
    for i in range(N):
        tmp = []
        for j in range(3):
            if Xgen[i,j] > 1:
                tmp.append(1.0)
            elif Xgen[i,j] < 0:
                tmp.append(0.0)
            else:
                tmp.append(Xgen[i,j])
        n_colors.append(tuple(tmp))
    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos,arrows=True,node_color=n_colors,with_labels=True)
    plt.savefig(f"GENERATED_graph{k}.png")
    plt.clf()
    k +=1
    #print("------------------------------------------------")