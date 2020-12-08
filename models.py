import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import re
import random
import time
import tensorflow_addons as tfa
from modules import *
from datetime import datetime
from dag_dataset import DAG_Dataset
from losses import *
from graph_visualizer import *

#***********************************************************
#                      Models                               
############################################################
class DAG_AE(tf.keras.Model):
    def __init__(self,args):
        super(DAG_AE,self).__init__(name="DAG_AE")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = BidirectionalDAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True)
        self.pred = CartesianProductClassifier()

    def call(self,inputs):
        X,A = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)

        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A])
        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        predictions = self.pred(hidden)
        return predictions

class Model2(tf.keras.Model):
    def __init__(self,args):
        super(Model2,self).__init__(name="Model2 - AutoEncoder")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = DAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True, return_Hi=True)
        self.pred = CartesianMLPDecoder()
        self.h_estimator = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(256, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(args.rnn_dim,"tanh")])

    def call(self,inputs):
        X,A = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)

        #encdode nodes in existing graph
        hidden_nodes, hidden_G = self.encoder([x_emb, A])
        #apply mask
        #hidden_nodes = tf.transpose(tf.ones([hidden_nodes.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden_nodes
        #hidden_G = tf.transpose(tf.ones([hidden_G.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden_G
        HG = tf.reduce_sum(hidden_nodes, axis=1,keepdims=True)

        concat = tf.concat([hidden_G, x_emb],axis=-1)
        #concat = tf.concat([tf.repeat(HG,x_emb.shape[1], axis=1), x_emb],axis=-1)
        estimated_h = self.h_estimator(concat)
        predictions = self.pred([hidden_nodes, estimated_h])
        return predictions, hidden_nodes, estimated_h

    def generate(self,inputs):
        X,A,new_node = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        x_hat_emb = self.node_embedding(new_node)
        #encdode nodes in existing graph
        hidden_nodes, hidden_G = self.encoder([x_emb, A])
        HG = tf.reduce_sum(hidden_nodes, axis=1,keepdims=True)
        #tf.print(HG)
        #tf.print(x_hat_emb)
        #tf.print(hidden_nodes.shape,HG.shape, x_hat_emb.shape)
        concat = tf.concat([hidden_G, tf.repeat(x_hat_emb,hidden_G.shape[1],axis=1)],axis=-1)
        estimated_h = self.h_estimator(concat)
        out = tf.concat([hidden_nodes, estimated_h],axis=-1)
        for l in self.pred._model:
            out = l(out)
        return out

class DAG_AE_newX(tf.keras.Model):
    def __init__(self,args):
        super(DAG_AE_newX,self).__init__(name="DAG_AE_newX")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = BidirectionalDAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True)
        self.dim_aligner = tf.keras.layers.Dense(args.rnn_dim)
        self.pred = EdgePredictor()

    def call(self,inputs,training=True):
        X,A,X_hat = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        x_hat_emb = self.node_embedding(X_hat)
        #encdode nodes in existing graph
        hidden= self.encoder([x_emb, A])
        #apply mask
        upsampled = self.dim_aligner(x_hat_emb)
        predictions = self.pred([hidden,upsampled],training=training)
        return predictions
    
    def generate(self,inputs):
        X,A,X_hat = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        x_hat_emb = self.node_embedding(X_hat)
        #encdode nodes in existing graph
        hidden= self.encoder([x_emb, A])
        upsampled = self.dim_aligner(x_hat_emb)
        predictions = self.pred([hidden,upsampled])
        return predictions

class DAG_VAE(tf.keras.Model):
    """
    Loss function balance (n_nodes^2)*reconstruction + z_dim*KLD - works for D15 and D50
    """
    def __init__(self,args):
        super(DAG_VAE,self).__init__(name="DAG_VAE")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = BidirectionalDAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True)
        self.z_mean = tf.keras.layers.Dense(args.rnn_dim,None)
        self.z_log_variance = tf.keras.layers.Dense(args.rnn_dim,None,kernel_initializer=tf.keras.initializers.Zeros())
        self.pred = CartesianMLPDecoder()
        self._z_dim = args.rnn_dim

    def call(self,inputs):
        X,A = inputs
  
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A])

        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        #get latent space parameters
        z_mean = self.z_mean(hidden)
        z_log_variance = self.z_log_variance(hidden)
        #sample z from normal distribution with given parameters
        z = tf.random.normal(shape=hidden.shape, mean=z_mean, stddev=tf.math.exp(z_log_variance/2))

        predictions = self.pred([hidden,z])
        return predictions,z_mean,z_log_variance

    def generate(self, inputs, num_new_nodes):
        X,A = inputs
  
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X,training=False)
        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A],training=False)

        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
    
        #sample z from normal distribution
        z = tf.random.normal(shape=[hidden.shape[0],num_new_nodes,self._z_dim], mean=0, stddev=1)
        predictions = self.pred([hidden,z],training=False)
        return predictions

############################################################
#                   Networks                               # 
############################################################

class DAG_RNNAutoencoder:
    def __init__(self, args):
        self._model = BiDAG_AE(args)  
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._loss = SigmoidFocalCrossEntropy()#
        self._metrics = {"loss":tf.keras.metrics.Mean(), "AUC_PR":tf.keras.metrics.AUC(curve="PR"), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self._z_dim = args.rnn_dim
        self.verbose = args.verbose
    
    def _kl_divergence(self, a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / (b_sd_squared1e-12)
        return (a_mean - b_mean) ** 2 / ((2 * b_sd_squared)+1e-12) + (ratio - tf.math.log(ratio) - 1) / 2

    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            self._model([X,A])
        with tf.GradientTape() as tape:
            predictions = self._model([X,A],training=True)
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a
            loss = self._loss(A,predictions)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
            #apply gradients
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
        #log metrics to tensorboard
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else:
                    metric(A, predictions)
                tf.summary.scalar("train/{}".format(name), metric.result())
            tf.summary.scalar("train/gradient_norm", gradient_norm)

            if self.verbose:
                #log gradients and weights
                for weights, grads in zip(self._model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
        return loss

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            loss = self.train_batch(X,A, mask_a)


    def train(self,dataset_train,dataset_dev,args):
        for epoch in range(args.epochs):
            self.train_epoch(dataset_train,args)
            metrics_eval = self.evaluate(dataset_dev, args)
            tf.print(f"Epoch {epoch+1}/{args.epochs}: {metrics_eval}")

    def predict_batch(self,X,A,mask_a=None):
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self._model.node_embedding(X,training=False)

        #encdode nodes in existing graph
        hidden = self._model.encoder([x_emb, A],training=False)
        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        #predict A_hat
        predictions = self._model.pred(hidden,training=False)
        predictions = predictions*mask_a
        return predictions, mask

    def evaluate_batch(self,X,A,mask_a):
        # Predict
        predictions, mask = self.predict_batch(X,A,mask_a)
        #compute loss 
        loss = self._loss(A, predictions)
        #compute metrics
        for name, metric in self._metrics.items():
            if name == "loss": metric(loss)
            else:
                metric(A, predictions)
        return predictions

    def evaluate(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            for metric in self._metrics.values():
                metric.reset_states()
            predictions = self.evaluate_batch(X,A, mask_a)

        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics
    
    def test(self, dataset, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            predictions = self.evaluate_batch(X,A,mask_a=mask_a)
        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics, predictions

class ModelType1:
    def __init__(self, args):
        self._model = DAG_AE_newX(args)
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.98, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=args.lr)
        self._loss = SigmoidFocalCrossEntropy()
        self._metrics = {"loss":tf.keras.metrics.Mean(), "AUC_PR":tf.keras.metrics.AUC(curve="PR"), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self.verbose = args.verbose

    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            self._model([X,A,X])
        with tf.GradientTape() as tape:
            predictions = self._model([X,A,X],training=True)
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a
            loss = self._loss(A, predictions)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
            #apply gradients
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
        #log metrics to tensorboard
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else:
                    metric(A, predictions)
                tf.summary.scalar("train/{}".format(name), metric.result())
            tf.summary.scalar("train/gradient_norm", gradient_norm)

            if self.verbose:
                #log gradients and weights
                for weights, grads in zip(self._model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
        return loss

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            loss = self.train_batch(X,A, mask_a)


    def train(self,dataset_train,dataset_dev,args):
        for epoch in range(args.epochs):
            self.train_epoch(dataset_train,args)
            metrics_eval = self.evaluate(dataset_dev, args)
            tf.print(f"Epoch {epoch+1}/{args.epochs}: {metrics_eval}")

    def predict_batch(self,X,A,mask_a=None):
        predictions = self._model([X,A,X],training=False)
        predictions = tf.transpose(predictions,perm=[0,2,1])
        return predictions

    def evaluate_batch(self,X,A,mask_a):
        # Predict
        predictions = self.predict_batch(X,A,mask_a)
        predictions = predictions*mask_a
        #compute loss 
        loss = self._loss(A, predictions)
        #compute metrics
        for name, metric in self._metrics.items():
            if name == "loss": metric(loss)
            else:
                metric(A, predictions)
        return predictions

    def evaluate(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            for metric in self._metrics.values():
                metric.reset_states()
            predictions = self.evaluate_batch(X,A, mask_a)

        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics
    
    def test(self, dataset, args,num_new_nodes=5):
        for metric in self._metrics.values():
            metric.reset_states()
        data = {"X":[], "A":[], "A_hat":[],"gen":[]}
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            predictions = self.evaluate_batch(X,A,mask_a=mask_a)
            gen = self.generate_sequentialy(X,A,[5 for k in range(num_new_nodes)],args.edge_threshold)
            for i in range(X.shape[0]):
                data["X"].append(X[i])
                data["A"].append(A[i])
                data["A_hat"].append(predictions[i])
                tmp = []
                data["gen"].append((gen[0][i],gen[1][i]))
        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)

        return metrics, data

    def generate(self, X, A, new_node):
        generated = self._model.generate([X,A,new_node])
        return generated
    
    def generate_sequentialy(self, X,A, new_nodes,threshold):
        for i in range(len(new_nodes)):
            generated  =  tf.expand_dims(self._model.generate([X,A,new_nodes[i]*tf.ones([X.shape[0],1])]),axis=2)
            generated = tf.where(generated >= threshold,1.,0.)
            X = tf.concat([X, new_nodes[i]*tf.ones([X.shape[0],1])],axis=1)
            A = tf.concat([A,tf.transpose(generated,perm=[0,2,1])],axis=1) #[n+1,n
            A = tf.concat([A,tf.zeros([A.shape[0],A.shape[1],1])],axis=2)
        return (X,A)

class ModelType2:
    def __init__(self, args):
        self._model = Model2(args)  
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.95, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._loss = SigmoidFocalCrossEntropy()
        self._loss2 = tf.keras.losses.Huber(delta=0.01)
        self._metrics = {"loss":tf.keras.metrics.Mean(), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self._z_dim = args.rnn_dim
        self.verbose = args.verbose


    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            self._model([X,A])
        with tf.GradientTape() as tape:
            predictions, hidden_nodes, h_estimate = self._model([X,A],training=True)
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a

    
            rec = tf.reduce_mean(self._loss(A, predictions))
            h_loss = tf.reduce_mean(self._loss2(hidden_nodes, h_estimate))
            loss = rec*A.shape[1]*A.shape[1] + A.shape[1]*h_loss*self._z_dim

            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
            #apply gradients
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
        #log metrics to tensorboard
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else:
                    metric(A, predictions)
                tf.summary.scalar("train/{}".format(name), metric.result())
            tf.summary.scalar("latent_loss", h_loss)
            tf.summary.scalar("train/gradient_norm", gradient_norm)

            if self.verbose:
                #log gradients and weights
                for weights, grads in zip(self._model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
        return loss

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            loss = self.train_batch(X,A, mask_a)

    def train(self,dataset_train,dataset_dev,args):
        for epoch in range(args.epochs):
            self.train_epoch(dataset_train,args)
            metrics_eval = self.evaluate(dataset_dev, args)
            tf.print(f"Epoch {epoch+1}/{args.epochs}: {metrics_eval}")

    def predict_batch(self,X,A,mask_a=None):
        predictions, hidden_nodes, h_estimate = self._model([X,A],training=True)
        #mask predictions using the mask_a input (set values out of the graph to zero)
        predictions = predictions*mask_a
        return predictions, hidden_nodes, h_estimate

    def evaluate_batch(self,X,A,mask_a):
        # Predict
        predictions, hidden_nodes, h_estimate = self.predict_batch(X,A,mask_a)
        #compute loss 
        
        rec = tf.reduce_mean(self._loss(A, predictions))
        h_loss = tf.reduce_mean(self._loss2(hidden_nodes, h_estimate))
        loss = A.shape[1]*A.shape[1]*rec + h_loss*self._z_dim

        for name, metric in self._metrics.items():
            if name == "loss": metric(loss)
            else:
                metric(A, predictions)
        return predictions*mask_a

    def evaluate(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            for metric in self._metrics.values():
                metric.reset_states()
            predictions = self.evaluate_batch(X,A, mask_a)

        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics
    
    def test(self, dataset, args,num_new_nodes=5):
        for metric in self._metrics.values():
            metric.reset_states()
        data = {"X":[], "A":[], "A_hat":[],"gen":[]}
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            predictions = self.evaluate_batch(X,A,mask_a=mask_a)
            gen = self.generate_sequentialy(X,A,[5 for k in range(num_new_nodes)],args.edge_threshold)
            for i in range(X.shape[0]):
                data["X"].append(X[i])
                data["A"].append(A[i])
                data["A_hat"].append(predictions[i])
                tmp = []
                data["gen"].append((gen[0][i],gen[1][i]))
        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)

        return metrics, data

    def generate(self, X, A, new_node):
        return self._model.generate([X,A,new_node])
    
    def generate_sequentialy(self, X,A, new_nodes,threshold):
        for i in range(len(new_nodes)):
            generated  = self._model.generate([X,A,new_nodes[i]*tf.ones([X.shape[0],1])])
            generated = tf.where(generated >= threshold,1.,0.)
            X = tf.concat([X, new_nodes[i]*tf.ones([X.shape[0],1])],axis=1)
            A = tf.concat([A,tf.transpose(generated,perm=[0,2,1])],axis=1) #[n+1,n
            A = tf.concat([A,tf.zeros([A.shape[0],A.shape[1],1])],axis=2)
        return (X,A)


class ModelType3:
    def __init__(self, args):
        self._model = DAG_VAE(args)
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.98, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._reconstruction_loss = SigmoidFocalCrossEntropy()
        self._metrics = {"loss":tf.keras.metrics.Mean(), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self._z_dim = args.rnn_dim
        self.verbose = args.verbose
    
    def kl_divergence(a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        epsilon = 1e-12
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / b_sd_squared+epsilon
        kld = (a_mean - b_mean) ** 2 / (2 * b_sd_squared+epsilon) + (ratio - tf.math.log(ratio) - 1) / 2
        return kld
        
    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            self._model([X,A])
        with tf.GradientTape() as tape:

            predictions,z_mean,z_log_variance = self._model([X,A],training=True)
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a
            #tf.debugging.assert_all_finite(predictions, "predictions masked_a incorrect")
            KLD = tf.reduce_mean(kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))
            #tf.debugging.assert_all_finite(KLD, "KLD incorrect")
            rec = tf.reduce_mean(self._reconstruction_loss(A, predictions))
            loss = 0.5*A.shape[1]*A.shape[1]*rec + KLD*self._z_dim         

            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
            #tf.debugging.assert_all_finite(gradients, "gradients not finite")
            #apply gradients
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
        #log metrics to tensorboard
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else:
                    metric(A, predictions)
                tf.summary.scalar("train/{}".format(name), metric.result())
            tf.summary.scalar("train/gradient_norm", gradient_norm)
            tf.summary.scalar("train/rec_loss", rec)
            tf.summary.scalar("train/laternt_loss",KLD)
            if self.verbose:
                #log gradients and weights
                for weights, grads in zip(self._model.trainable_weights, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads)
        return loss

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            #print(mask_a.shape)
            loss = self.train_batch(X,A, mask_a)


    def train(self,dataset_train,dataset_dev,args):
        for epoch in range(args.epochs):
            self.train_epoch(dataset_train,args)
            metrics_eval = self.evaluate(dataset_dev, args)
            tf.print(f"Epoch {epoch+1}/{args.epochs}: {metrics_eval}")

    def predict_batch(self,X,A,mask_a=None):
        predictions,mu,sigma = self._model([X,A],training=True)
        predictions = predictions*mask_a
        return predictions, mu,sigma

    def evaluate_batch(self,X,A,mask_a):
        predictions,z_mean,z_log_variance = self._model([X,A],training=True)
        #mask predictions using the mask_a input (set values out of the graph to zero)
        predictions = predictions*mask_a

        KLD = tf.reduce_mean(kl_divergence(z_mean, tf.math.exp(z_log_variance/2), 0,1))
        rec = tf.reduce_mean(self._reconstruction_loss(A, predictions))
        loss = 0.5*A.shape[1]*A.shape[1]*rec + KLD*self._z_dim 
        #compute metrics
        for name, metric in self._metrics.items():
            if name == "loss": metric(loss)
            else:
                metric(A, predictions)
        return predictions

    def evaluate(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            for metric in self._metrics.values():
                metric.reset_states()
            predictions = self.evaluate_batch(X,A, mask_a)

        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics
    
    def test(self, dataset, args, num_new_nodes=5):
        for metric in self._metrics.values():
            metric.reset_states()
        data = {"X":[], "A":[], "A_hat":[],"gen":[]}
        for batch in dataset.batches(args.batch_size):
            X = batch["X"]
            A = batch["A"]
            mask_a = batch["mask_a"]
            predictions = self.evaluate_batch(X,A,mask_a=mask_a)
            gen = self.generate_sequentialy(X,A,[5 for k in range(num_new_nodes)],args.edge_threshold)
            for i in range(X.shape[0]):
                data["X"].append(X[i])
                data["A"].append(A[i])
                data["A_hat"].append(predictions[i])
                data["gen"].append((gen[0][i],gen[1][i]))
        metrics = {name: float(metric.result()) for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format("dev", name), value)
        return metrics, data

    def generate(self, X, A, num_new_nodes):
        # Embedd input nodes
        generated = self._model.generate([X,A],num_new_nodes)
        return generated

    def generate_sequentialy(self, X,A, new_nodes,threshold, max_steps =100):
        counter = 0
        steps = 0
        while counter < len(new_nodes) and steps < max_steps:
            steps +=1
            tmp = self._model.generate([X,A],1)
            if len(tmp.shape) == 1:
                tmp = tf.expand_dims(tmp,axis=0)
            generated  = tf.expand_dims(tmp,axis=2)
            generated = tf.where(generated >= threshold,1.,0.)
            if tf.reduce_sum(generated) > 0:
                X = tf.concat([X, new_nodes[counter]*tf.ones([X.shape[0],1])],axis=1)
                A = tf.concat([A,tf.transpose(generated,perm=[0,2,1])],axis=1) #[n+1,n
                A = tf.concat([A,tf.zeros([A.shape[0],A.shape[1],1])],axis=2)
                counter += 1
        return (X,A)

def generate_honeyuser_location(filename, model, args, num_nodes):
    with open(filename, "rb") as f:
        tmp = pickle.load(f)
    X = tmp["X"]
    A = tmp["A"]
    #tf.print(X,output_stream="file://out.txt")
    _,newA = network.generate_sequentialy(tf.cast(np.expand_dims(X,axis=0),tf.float32), tf.cast(np.expand_dims(A,axis=0),tf.float32), 5*tf.ones([new_nodes,1]), args.edge_threshold)
    #tf.print(newA,output_stream="file://out.txt", summarize=-1)
    newA = newA[0]
    has_ou = 0
    for i in range(new_nodes):
        edges = tf.where(newA[:,i] >= args.edge_threshold)
        for e in edges:
            e = int(tf.squeeze(e))
            if e < A.shape[0]:
                if tmp["objects"][tmp["node_id_to_oid"][e]]["objectType"] == "ou":
                    has_ou += 1

                #if tmp["objects"][tmp["node_id_to_oid"][e]]["objectType"] != "user":
                tf.print(tmp["objects"][tmp["node_id_to_oid"][e]]["dn"],output_stream="file://output_generated.txt")
        tf.print("-------------------",output_stream="file://output_generated.txt")
    print(has_ou/num_nodes)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    #global parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    #encoder parameters
    parser.add_argument("--feature_dim", default=6, type=int, help="Feature dimension")
    parser.add_argument("--rnn_dim", default=64, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--node_emb_dim", default=6, type=int, help="Graph embedding dimension")
    parser.add_argument("--edge_threshold", default=0.35 ,type=float, help="Edge limit value during generation")
    parser.add_argument("--verbose", default=False, type=bool, help="Verbose output")
    parser.add_argument("--model", default=3, type=int, help="Verbose output")
    args = parser.parse_args()
    args = parser.parse_args()
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    #set amount of cores to use
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    train_data_150 = "./dag-data/150_nodes/train_150.pickle"
    dev_data_150 = "./dag-data/150_nodes/dev_150.pickle"

    train_data_50 = "./dag-data/50_nodes/train_50.pickle"
    dev_data_50 = "./dag-data/50_nodes/dev_50.pickle"

    train_data_15 = "./dag-data/15_nodes/train_15.pickle"
    dev_data_15 = "./dag-data/15_nodes/dev_15.pickle"

    train_data_500 = "./dag-data/500_nodes/train_500.pickle"
    dev_data_500 = "./dag-data/500_nodes/dev_500.pickle"
 
    models = {3:ModelType3(args),2:ModelType2(args)}
    #models = {1:ModelType1(args)}
    """
    #train 15
    args.batch_size=32
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42,max_samples=1000)
    dataset_train.read_nx_pickle(train_data_150)
    dataset_dev = DAG_Dataset(shuffle_batches=False,seed=42,max_samples=200)
    dataset_dev.read_nx_pickle(dev_data_150)
    for i in models.keys():
        models[i].train(dataset_train, dataset_dev, args)
        models[i]._model.save_weights(f"./models/Type{i}/model_after150.h5", overwrite=True)
        #metrics, data = models[i].test(dataset_dev,args)
        #with open(f"./Experiments/Type{i}/m={i}_rdim{args.rnn_dim}_15_comparison.pickle", "wb") as outf:
        #   pickle.dump(data, outf)
    """
    """

    #train 50
    args.batch_size=10
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42,max_samples=1)
    dataset_train.read_nx_pickle(train_data_15)
    dataset_dev = DAG_Dataset(shuffle_batches=False,seed=42,max_samples=500)
    dataset_dev.read_nx_pickle(dev_data_500)

    edge_t_levels = [0.00001 ,0.001, 0.1,0.25, 0.5]
    for i in models.keys():
        models[i].train(dataset_train, dataset_train, args)
        #models[i]._model.save_weights(f"./models/Type{i}/model_after50.h5", overwrite=True)
        models[i]._model.load_weights(f"./models/Type{i}/model_after150.h5",skip_mismatch=False)
        for e in range(len(edge_t_levels)):
            args.edge_threshold = edge_t_levels[e]
            metrics, data = models[i].test(dataset_dev,args,num_new_nodes=15)
            #print(data.keys())
            DNR, MUNEC = check_connectivity(data)
            vr = validity_ratio(data)
            print(f"Model{i},dataset=500,e={edge_t_levels[e]},DNR={DNR},VR={vr} ,MUNEC={MUNEC}")
            with open(f"./Experiments/Type{i}/m={i}_rdim{args.rnn_dim}_500_generation_e={edge_t_levels[e]}_model150.pickle", "wb") as outf:
                pickle.dump(data, outf)
        print("--------------------------------------------------")
    """



    with open("./preprocessed_for_dagrnn.pickle", "rb") as f:
        tmp = pickle.load(f)
    X = tmp["X"]
    A = tmp["A"]

    args.batch_size=1
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42,max_samples=1)
    dataset_train.read_nx_pickle(train_data_15)
    network = models[3]
    args.edge_threshold = 0.1
    new_nodes = 30
    network.train(dataset_train, dataset_train, args)
    network._model.load_weights(f"./models/Type3/model_after50.h5",skip_mismatch=False)
    

    generate_honeyuser_location("./preprocessed_for_dagrnn.pickle", network, args, new_nodes)

    """
    #train 150
    args.batch_size=32
    args.edge_threshold=0.3
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42,max_samples=1000)
    dataset_train.read_nx_pickle(train_data_150)
    dataset_dev = DAG_Dataset(shuffle_batches=False,seed=42,max_samples=200)
    dataset_dev.read_nx_pickle(dev_data_150)
    for i in models.keys():
        models[i].train(dataset_train, dataset_dev, args)
        models[i]._model.save_weights(f"./models/Type{i}/model_after150.h5", overwrite=True)
        metrics, data = models[i].test(dataset_dev,args)
        with open(f"./Experiments/Type{i}/m={i}_rdim{args.rnn_dim}_150_generation.pickle", "wb") as outf:
            pickle.dump(data, outf)

    
    "
    args.batch_size=10
    args.edge_threshold=0.25
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42,max_samples=750)
    dataset_train.read_nx_pickle(train_data_500)
    dataset_dev = DAG_Dataset(shuffle_batches=False,seed=42,max_samples=150)
    dataset_dev.read_nx_pickle(dev_data_500)
    for i in models.keys():
        models[i].train(dataset_train, dataset_dev, args)
        models[i]._model.save_weights(f"./models/Type{i}/model_after500.h5", overwrite=True)
        metrics, data = models[i].test(dataset_dev,args)
        with open(f"./Experiments/Type{i}/m={i}_rdim{args.rnn_dim}_500_generation.pickle", "wb") as outf:
            pickle.dump(data, outf)
 
    #train 15
    args.batch_size=256
    dataset_train = DAG_Dataset(shuffle_batches=True,seed=42)
    dataset_train.read_nx_pickle(train_data_15)
    dataset_dev = DAG_Dataset(shuffle_batches=False,seed=42,max_samples=500)
    dataset_dev.read_nx_pickle(dev_data_15)
    for i in models.keys():
        models[i].train(dataset_train, dataset_dev, args)
        models[i]._model.save_weights(f"./models/Type{i}/model_after15.h5", overwrite=True)
        models[i]._model.load_weights(f"./models/Type{i}/model_after15.h5",by_name=False,skip_mismatch=False)
        metrics, data = models[i].test(dataset_dev,args)
        with open(f"./Experiments/Type{i}/m={i}_rdim{args.rnn_dim}_15_generation.pickle", "wb") as outf:
            pickle.dump(data, outf)
    """