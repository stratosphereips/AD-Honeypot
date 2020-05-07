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

class DAGDataset():

    def __init__(self,shuffle_batches=False):
        self._size = 0
        self._data = {"features":[], "conditions":[]}
        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    def read_nx_pickle(self, filename):
        with open(filename, "rb") as f:
            in_data = pickle.load(f)
        for sample in in_data:
            G = nx.from_numpy_array(sparse.csr.csr_matrix.todense(sample[1]), create_using=nx.DiGraph)
            a = sparse.csr.csr_matrix.todense(sample[1]).transpose()
            x = tf.one_hot(sample[0],7).numpy()
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
    def __init__(self,args):
        self._z_dim = args.z_dim
        self._feature_dim = args.feature_dim
        self._optimizer = tf.optimizers.Adam(clipnorm=True)


        #ENCODER 
        input_A = tf.keras.layers.Input(shape=(None,None),name="Input_A")
        input_X = tf.keras.layers.Input(shape=(None,args.feature_dim), name="Input_X")
        z = DynamicRNNEncoder(rnn_dim=args.enc_rnn_dim, agg_hidden_size=args.agg_hidden, z_dim=args.z_dim)(input_X,input_A) 
        self.encoder = tf.keras.Model([input_X,input_A],[z],name="Encoder")

        
        #DECODER
        input_z = tf.keras.layers.Input(shape=(args.z_dim),name="Input_z")  

        means,F,A,N= DynamicRNNDecoder(rnn_dim=args.dec_rnn_dim,z_dim=args.z_dim, feature_dim=args.feature_dim,
            modul_hidden_size=args.module_hidden,agg_hidden_size=args.agg_hidden)(input_z)
        self.decoder = tf.keras.Model(input_z,[means,F,A,N],name="Decoder")


        #losses and metrics
        self._size_loss = tf.keras.losses.Poisson()
        self._node_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
        self._structural_loss = tf.keras.losses.BinaryCrossentropy()
        self._latent_loss = None
        self._metrics = {
            'loss': tf.metrics.Mean(),
            'loss_size': tf.keras.metrics.Mean(),
            "loss_features":tf.metrics.Mean(),
            "loss_structure":tf.metrics.Mean(),

        }
        self._loss_fn = tf.keras.losses.Poisson()
        
        #logging
        self._writer = tf.summary.create_file_writer(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", flush_millis=10 * 1000)
        
        #Show model summary
        self.encoder.summary()
        self.decoder.summary()

    """
    def combined_loss(self,golden, model_output,mask):
        means, F_hat,A_hat = model_output
        max_node = A_hat.shape[1]
        num_nodes, F, A = golden


        

        if A_hat.shape[1] < A.shape[1]:
            A_hat = tf.pad(A_hat,tf.constant([[0,0],[0,A.shape[1]-A_hat.shape[1]],[0,A.shape[2]-A_hat.shape[2]]]),"CONSTANT")
            F_hat = tf.pad(F_hat,tf.constant([[0,0],[0,F.shape[1]-F_hat.shape[1]],[0,F.shape[2]-F_hat.shape[2]]]),"CONSTANT")

        elif A_hat.shape[1] > A.shape[1]:
            A = tf.pad(A,tf.constant([[0,0],[0,A_hat.shape[1]-A.shape[1]],[0,A_hat.shape[2]-A.shape[2]]]),"CONSTANT")
            F = tf.pad(F,tf.constant([[0,0],[0,F_hat.shape[1]-F.shape[1]],[0,F_hat.shape[2]-F.shape[2]]]),"CONSTANT")
      
        loss_p = tf.math.reduce_mean(tf.keras.losses.poisson(np.expand_dims(num_nodes,axis=1),means))
        loss_s =  tf.math.reduce_mean(tf.keras.losses.mean_squared_error(A,A_hat))
        #loss_f = self.feature_loss(F,F_hat,mask)
        loss_f = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(F,F_hat))
        return tf.math.reduce_sum(loss_p + loss_s + loss_f)
        
    

    def pad_same_sizes(self,X,Y,mask=None):
        if X.shape[1] < Y.shape[1]:
            X = tf.pad(X,tf.constant([[0,0],[0,Y.shape[1]-X.shape[1]],[0,Y.shape[2]-X.shape[2]]]),"CONSTANT")
        elif X.shape[1] > Y.shape[1]:
            Y = tf.pad(Y,tf.constant([[0,0],[0,X.shape[1]-Y.shape[1]],[0,X.shape[2]-Y.shape[2]]]),"CONSTANT")

        return X,Y,mask
    
    def structural_loss(self,A,A_hat):
        #print("mse shape",tf.keras.losses.mean_squared_error(A,A_hat).shape)
        mse = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(A,A_hat))
        return mse

    def feature_loss(self, F, F_hat,mask):

        mse = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(F,F_hat))
        #print(mse.shape)
        #print(tf.boolean_mask(mse, [mask]).shape)
        return mse


    """

    #@tf.function
    def align_feature_tensors(self,F,F_hat,n,n_hat):
        max_nodes = tf.math.reduce_max([n,n_hat])
        mask_len = tf.math.reduce_max([n,n_hat],axis=0)
        masks = tf.sequence_mask(mask_len)
        #print(f"max_nodes:{max_nodes}")
        #print(f"F:{F.shape}, F_hat:{F_hat.shape}")
        F = tf.keras.preprocessing.sequence.pad_sequences(F,maxlen=max_nodes,padding="post",value=tf.zeros(self._feature_dim),dtype="float32")
        F_hat = tf.keras.preprocessing.sequence.pad_sequences(F_hat,maxlen=max_nodes,padding="post",value=tf.zeros(self._feature_dim),dtype="float32")
        #F = tf.convert_to_tensor(F)
        #F._keras_mask = masks
        #F_hat._keras_mask = masks
        #F = tf.cast(F,dtype=tf.float32)
        #F_hat = tf.cast(F_hat,dtype=tf.float32)
        #print(f"F:{F.shape}, F_hat:{F_hat.shape}")
        #print(f"MSE:{self._node_loss(F,F_hat,masks)}")
        return F,F_hat,masks
    
    #@tf.function
    def align_structure_tensors(self,A,A_hat,n,n_hat):
        max_nodes = tf.math.reduce_max([n,n_hat])
        mask_len = tf.math.reduce_max([n,n_hat],axis=0)
        #print(f"A:{A.shape}, A_hat:{A_hat.shape}")
        if A.shape[1] < A_hat.shape[1]:
            diff = A_hat.shape[1] - A.shape[1]
            A = tf.pad(A,[[0,0],[0,diff],[0,diff]])
        if A.shape[1] > A_hat.shape[1]:
            diff = A.shape[1] - A_hat.shape[1]
            A_hat = tf.pad(A_hat,[[0,0],[0,diff],[0,diff]])
        #print(f"A:{A.shape}, A_hat:{A_hat.shape}")
        mask_list = [tf.pad(tf.ones([i,i]),[[0,max_nodes-i],[0,max_nodes-i]]) for i in mask_len]
        masks = tf.convert_to_tensor(mask_list)
        masks = tf.cast(masks, dtype=tf.bool)
        #print("Mask:",masks.shape)
        #masks = tf.sequence_mask(mask_len) # TODO FIX THIS!!!!!
        return A, A_hat, masks

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
        
            F_fixed,F_hat_fixed,F_mask_fixed = self.align_feature_tensors(F, F_hat, gold_N, n_count)
            A_fixed,A_hat_fixed,A_mask_fixed = self.align_structure_tensors(A, A_hat,gold_N, n_count)

            #set all values of padding to 0
            A_fixed = tf.where(A_mask_fixed, A_fixed, 0)
            A_hat_fixed = tf.where(A_mask_fixed, A_hat_fixed, 0)
            #compute losses
            structural_loss = self._structural_loss(A_fixed,A_hat_fixed)
            feature_loss = self._node_loss(F_fixed,F_hat_fixed)
            size_loss = self._size_loss(np.expand_dims(gold_N,axis=1),means)
            latent_loss = 0 # tf.reduce_mean(self._kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))


            #print(f"N:{size_loss}, F:{feature_loss}, S:{structural_loss}, L:{latent_loss}")
            max_nodes = tf.cast(tf.math.reduce_max([gold_N,n_count]),tf.float32)
            #combine them
            loss = size_loss + feature_loss + structural_loss + self._z_dim*latent_loss
            
            #make step
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)

            #log metrics
            with self._writer.as_default():
                tf.summary.scalar("training/structural_loss", structural_loss)
                tf.summary.scalar("training/feature_loss", feature_loss)
                tf.summary.scalar("training/size_loss", size_loss)
                tf.summary.scalar("training/latent_loss", latent_loss)
                #tf.summary.scalar("training/loss", loss)
            return loss

    def train(self, train_data, dev_data, epochs, batch_size):
        for epoch in range(0,epochs):
            for batch in train_data.batches(batch_size):
                self.train_batch(batch)
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, batch_size)
            print(f"EPOCH {epoch}/{epochs}:\tDev loss:{metrics['loss']}, S:{metrics['loss_size']}, F:{metrics['loss_features']}, A:{metrics['loss_structure']}")
    
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
    
        F_fixed,F_hat_fixed,F_mask_fixed = self.align_feature_tensors(F, F_hat, gold_N, n_count)
        A_fixed,A_hat_fixed,A_mask_fixed = self.align_structure_tensors(A, A_hat,gold_N, n_count)

        #set all values of padding to 0
        A_fixed = tf.where(A_mask_fixed, A_fixed, 0)
        A_hat_fixed = tf.where(A_mask_fixed, A_hat_fixed, 0)
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

    def generate(self, count):
        def sample_z(batch_size):
            return tf.random.normal(shape=[batch_size, self._z_dim])
        z = sample_z(count)
        means, F_hat, A_hat, n_count = self.decoder(z)
        tf.print(z)
        tf.print("nodes:",n_count)
        print("---------------------")
        tf.print(F_hat)
        print("---------------------")
        tf.print(A_hat)
        with self._writer.as_default():
            tf.summary.histogram("A_hat",A_hat)
        return F_hat, A_hat

    def test(self,batch):
        gold_N = batch[1]
        F = batch[0][0]
        A = batch[0][1]
        z_mean, z_log_variance = self.encoder([F,A],training=False)
        z = tf.random.normal(shape=[self._z_dim], mean=z_mean, stddev=tf.math.exp(z_log_variance/2))
        means, F_hat, A_hat, n_count = self.decoder(z, training=False)
        A_hat = tf.where(A_hat > 0.2,1,0)
        return F_hat,A_hat



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--feature_dim", default=7, type=int, help="Feature dimension")
    parser.add_argument("--enc_rnn_dim", default=64, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--z_dim", default=64, type=int, help="Graph embedding dimension")

    parser.add_argument("--dec_rnn_dim", default=64, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--agg_hidden", default=64, type=int, help="Dimenion of Aggregators hidden layers")
    parser.add_argument("--module_hidden", default=64, type=int, help="Dimension of Decoder Modules hidden layers")
    parser.add_argument("--edge_threshold", default=0.5, type=int, help="Dimension of Decoder Modules hidden layers")
    args = parser.parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    train_data = "./dag-data/DAG_small_generated/train/train_small_dag.pickle"
    dev_data = "./dag-data/DAG_small_generated/dev/dev_small_dag.pickle"

    dataset_train = DAGDataset()
    dataset_train.read_nx_pickle(train_data)
    dataset_dev = DAGDataset()
    dataset_dev.read_nx_pickle(dev_data)
    

    network = Network(args)

    print("Build complete")
    

    network.train(dataset_train, dataset_dev, args.epochs, args.batch_size)
    #network.save('./models/vae.')
    #network.save_weights('./checkpoints/vae.')
    #network.generate(1)
    data = dataset_dev.batches(10)
    batch = next(data)
    F,A = network.test(batch)
    #print(np.argmax(F[0],axis=1))
    #print("--------------")
    #print(np.argmax(F[1],axis=1))
    #print("--------------")
    #print(A[1])
    #print("##############")
    #print(np.argmax(F[2],axis=1))
    #print("--------------")
    #print(A[2])
    #print("##############")
    output = {"features":F, "structure":A}
    with open('dev_output_ae.pickle', 'wb') as handle:
        pickle.dump(output, handle)