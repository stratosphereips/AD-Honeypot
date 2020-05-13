import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import sys
from scipy import sparse
from dynamic_rnn import DynamicRNNEncoder,DynamicRNNDecoder, MLP

import tensorflow as tf
import numpy as np
import collections
import recurrent_fixed
from datetime import datetime
def frobenius_loss(X,Y):
    norm = tf.norm(X-Y,ord='fro',axis=(1,2))
    #tf.print("NORM:")
    #tf.print(norm)
    return tf.math.reduce_mean(norm)

class DAGDataset():

    def __init__(self,shuffle_batches=False,seed=42):
        self._size = 0
        self._data = {"features":[], "conditions":[]}
        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
    """
    def transform_graph(self,A, ordering):
        new_A = np.zeros_like(A)
        mapping = {ordering[i]:i for i in range(len(ordering))}
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                new_A[mapping[i],mapping[j]] = A[i,j]
        print(new_A)
    """
    def read_nx_pickle(self, filename):
        with open(filename, "rb") as f:
            in_data = pickle.load(f)
        for sample in in_data:
            #G = nx.from_numpy_array(sparse.csr.csr_matrix.todense(sample[1]), create_using=nx.DiGraph)
            a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
            x = tf.one_hot(sample[0],7).numpy()
            self._data["features"].append(x)
            self._data["conditions"].append(a)
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
                        #print(A)
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
    def __init__(self,args):
        self._z_dim = args.z_dim
        self._feature_dim = args.feature_dim
        self._edge_threshold = args.edge_threshold
        self._optimizer = tf.optimizers.Adam(learning_rate=0.01)


        #ENCODER 
        input_A = tf.keras.layers.Input(shape=(None,None),name="Input_A")
        input_X = tf.keras.layers.Input(shape=(None,args.feature_dim), name="Input_X")
        z = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, agg_hidden_size=args.agg_hidden, z_dim=args.z_dim)(input_X,input_A) 
        self.encoder = tf.keras.Model([input_X,input_A],[z],name="Encoder")

        
        #DECODER
        input_z = tf.keras.layers.Input(shape=(args.z_dim),name="Input_z")  

        means,F,A,N= DynamicRNNDecoder(rnn_dim=args.dec_rnn_dim,z_dim=args.z_dim, feature_dim=args.feature_dim,
            modul_hidden_size=args.module_hidden,agg_hidden_size=args.agg_hidden, edge_treshold=args.edge_threshold)(input_z)
        self.decoder = tf.keras.Model(input_z,[means,F,A,N],name="Decoder")


        #losses and metrics
        self._size_loss = tf.keras.losses.Poisson()
        self._node_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
        self._structural_loss = frobenius_loss
        self._latent_loss = None
        self._metrics = {
            'loss': tf.metrics.Mean(),
            'loss_size': tf.keras.metrics.Mean(),
            "loss_features":tf.metrics.Mean(),
            "loss_structure":tf.metrics.Mean(),

        }
        
        #logging
        self._writer = tf.summary.create_file_writer(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", flush_millis=10 * 1000)
        
        #Show model summary
        #self.encoder.summary()
        #self.decoder.summary()

    #@tf.function(experimental_relax_shapes=True)
    def align_feature_tensors(self,F,F_hat,n,n_hat):
        #tf.print(n)
        #tf.print("n^:")
        #tf.print(n_hat)
        #tf.print("---------------------------------")
        max_nodes = tf.math.maximum(F.shape[1], F_hat.shape[1])

        mask_gold = tf.sequence_mask(tf.cast(n,tf.int32),maxlen=max_nodes)
        mask_gold = tf.stack([mask_gold for k in range(F.shape[2])],axis=2)
        
        mask_pred = tf.sequence_mask(n_hat,maxlen=max_nodes)
        mask_pred = tf.stack([mask_pred for k in range(F_hat.shape[2])],axis=2)

        
        #padd tensors to same size
        F_padded = tf.pad(F,[[0,0],[0,max_nodes-F.shape[1]],[0,0]])
        F_hat_padded = tf.pad(F_hat,[[0,0],[0,max_nodes-F_hat.shape[1]],[0,0]])
        

        #tf.print(mask_gold)
        #tf.print("---------------------------------")
        #tf.print(mask_pred)
        #tf.print("---------------------------------")
        #tf.print("F^:")
        #tf.print(F_hat_padded)
        #tf.print("---------------------------------")
        #tf.print("F:")
        #tf.print(F_padded)
        #tf.print("---------------------------------")
        #tf.print("Features unmasked:", self._node_loss(F_padded,F_hat_padded))
        #Apply masks
        F_hat_padded = tf.where(mask_pred,F_hat_padded,0.0)
        F_padded = tf.where(mask_gold,F_padded,0.0)
        #tf.print("Features masked:", self._node_loss(F_padded,F_hat_padded))
        #tf.print("F hat padded&masked")
        #tf.print(F_hat_padded)
        #tf.print("---------------------------------")
        #tf.print("F padded&masked")
        #tf.print(F_padded)
        #tf.print("###########################################")
        return F_padded,F_hat_padded,mask_gold
    
    #@tf.function(experimental_relax_shapes=True)
    def align_structure_tensors(self,A,A_hat,n,n_hat):
        max_nodes = tf.math.maximum(A.shape[1], A_hat.shape[1])
        n = tf.cast(n,tf.int32)
        #create masls
        mask_gold = tf.convert_to_tensor([tf.pad(tf.ones([n[i],n[i]]),[[0,max_nodes-n[i]],[0,max_nodes-n[i]]]) for i in range(A.shape[0])]) 
        mask_pred = tf.convert_to_tensor([tf.pad(tf.ones([n_hat[i],n_hat[i]]),[[0,max_nodes-n_hat[i]],[0,max_nodes-n_hat[i]]]) for i in range(A.shape[0])])
        #pad tensors
        A_padded = tf.pad(A,[[0,0],[0,max_nodes-A.shape[1]],[0,max_nodes-A.shape[2]]])
        A_hat_padded = tf.pad(A_hat,[[0,0],[0,max_nodes-A_hat.shape[1]],[0,max_nodes-A_hat.shape[2]]])
        

        """
        tf.print("n:",n)
        tf.print("n^:", n_hat)
        tf.print(mask_gold)
        tf.print("-------------------------")
        tf.print(mask_pred)
        tf.print("-------------------------")
        tf.print("A")
        tf.print(A_padded)
        tf.print("-------------------------")
        tf.print("A^")
        tf.print(A_hat_padded)
        tf.print("-------------------------")
        """

        #tf.print("Structural loss unmasked:", self._size_loss(A_padded,A_hat_padded))

        #apply masks
        A_padded = A_padded*mask_gold
        A_hat_padded = A_hat_padded*mask_pred

        #tf.print("Structural loss masked:", self._size_loss(A_padded,A_hat_padded))
        """
        tf.print("A masked:")
        tf.print(A_padded)
        tf.print("-------------------------")
        tf.print("A^ masked")
        tf.print(A_hat_padded)
        tf.print("###########################")
        """
        return A_padded, A_hat_padded,mask_gold

    def _kl_divergence(self, a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / b_sd_squared
        return (a_mean - b_mean) ** 2 / (2 * b_sd_squared) + (ratio - tf.math.log(ratio) - 1) / 2

    #@tf.function
    def train_batch(self, batch):
        gold_N = batch[1]
        F = batch[0][0]
        A = batch[0][1]
        with tf.GradientTape() as tape:
            #encode graphs into latent space
            #z = self.encoder([F,A],training=True)
            z = self.encoder([F,A],training=True)

            #decode 
            means, F_hat, A_hat, n_count = self.decoder(z, training=True)

            #create masks for node features
            F_mask = tf.sequence_mask(n_count)
            
            max_nodes = tf.math.reduce_max(n_count)
            
            F_fixed,F_hat_fixed,F_mask_fixed = self.align_feature_tensors(F, F_hat, gold_N, n_count)
            A_fixed,A_hat_fixed,A_mask_fixed = self.align_structure_tensors(A, A_hat,gold_N, n_count)

            #compute losses
            structural_loss = self._structural_loss(A_fixed,A_hat_fixed)
            feature_loss = self._node_loss(F_fixed,F_hat_fixed)
            size_loss = self._size_loss(np.expand_dims(gold_N,axis=1),means)
            latent_loss = 0 # tf.reduce_mean(self._kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))


            #combine them
            loss = size_loss + feature_loss + structural_loss + self._z_dim*latent_loss
            #make step
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 0.25)


            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)

            #log metrics
            with self._writer.as_default():
                tf.summary.scalar("training/structural_loss", structural_loss)
                tf.summary.scalar("training/feature_loss", feature_loss)
                tf.summary.scalar("training/size_loss", size_loss)
                tf.summary.scalar("training/latent_loss", latent_loss)
                #tf.summary.scalar("training/loss", loss)
                #tf.summary.scalar("training/gradient_norm", gradient_norm)
            return loss

    def train(self, train_data, dev_data, epochs, batch_size):
        for epoch in range(0,epochs):
            for batch in train_data.batches(batch_size):
                self.train_batch(batch)
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, batch_size)
            tf.print(f"EPOCH {epoch+1}/{epochs}:\tDev loss:{metrics['loss']}, S:{metrics['loss_size']}, F:{metrics['loss_features']}, A:{metrics['loss_structure']}")
    
    def evaluate_batch(self, batch):
        gold_N = batch[1]
        F = batch[0][0]
        A = batch[0][1]
        #encode graphs into latent space
        z = self.encoder([F,A],training=False)

        #tf.print(z)
        #decode 
        means, F_hat, A_hat, n_count = self.decoder(z, training=False)

        #create masks for node features
        F_mask = tf.sequence_mask(n_count)
        max_nodes = tf.math.reduce_max(n_count)


        F_fixed,F_hat_fixed,F_mask_fixed = self.align_feature_tensors(F, F_hat, gold_N, n_count)
        A_fixed,A_hat_fixed,A_mask_fixed = self.align_structure_tensors(A, A_hat,gold_N, n_count)

        #compute losses
        structural_loss = self._structural_loss(A_fixed,A_hat_fixed)
        feature_loss = self._node_loss(F_fixed,F_hat_fixed)
        size_loss = self._size_loss(np.expand_dims(gold_N,axis=1),means)
        latent_loss = 0#tf.reduce_mean(self._kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))


        #print(f"N:{size_loss}, F:{feature_loss}, S:{structural_loss}, L:{latent_loss}")

        #combine them
        max_nodes = tf.cast(tf.math.reduce_max([gold_N,n_count]),tf.float32)
        loss = size_loss + feature_loss + structural_loss + self._z_dim*latent_loss        
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            elif name == "loss_size":
                metric(size_loss)
            elif name == "loss_features":
                metric(feature_loss)
            elif name == "loss_structure":
                metric(structural_loss)
            elif name == "loss_latent":
                metric(latent_loss)
    
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

    def test(self,batch):
        gold_N = batch[1]
        F = batch[0][0]
        A = batch[0][1]
        z = self.encoder([F,A],training=False)
        means, F_hat, A_hat, n_count = self.decoder(z, training=False)
        return F_hat,A_hat,n_count

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--feature_dim", default=7, type=int, help="Feature dimension")
    parser.add_argument("--enc_rnn_dim", default=16, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--z_dim", default=12, type=int, help="Graph embedding dimension")

    parser.add_argument("--dec_rnn_dim", default=16, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--agg_hidden", default=32, type=int, help="Dimenion of Aggregators hidden layers")
    parser.add_argument("--module_hidden", default=32, type=int, help="Dimension of Decoder Modules hidden layers")
    parser.add_argument("--edge_threshold", default=0.5, type=float, help="Edge limit value during generation")
    args = parser.parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    train_data = "./dag-data/dummy_dataset/train/train_dummy.pickle"
    dev_data = "./dag-data/dummy_dataset/dev/dev_dummy.pickle"

    dataset_train = DAGDataset(shuffle_batches=True)
    dataset_train.read_nx_pickle(train_data)
    dataset_dev = DAGDataset(shuffle_batches=True, seed=42)
    dataset_dev.read_nx_pickle(dev_data)
    dataset_test = DAGDataset()
    dataset_test.read_nx_pickle("./dag-data/dummy.pickle")
    network = Network(args)

    #print("Build complete")
    

    network.train(dataset_train, dataset_dev,args.epochs,args.batch_size)
    #network.save('./models/vae.')
    #network.save_weights('./checkpoints/vae.')
    #network.generate(1)
    
    count = 5
    data = dataset_dev.batches(count)
    batch = next(data)
    F,A,n = network.test(batch)
    #   print(batch)
    for i in range(count):
        tf.print("Original A:")
        tf.print(batch[0][1][i])
        tf.print("---------------")
        tf.print("predicted A")
        tf.print(A[i,:,:])
        tf.print("---------------")
        tf.print("original F:")
        tf.print(np.argmax(batch[0][0][i], axis=1))
        tf.print("---------------")
        tf.print("predicted F")
        tf.print(np.argmax(F[i,:,:], axis=1))
        tf.print("---------------")
    tf.print("Sizes:",n)