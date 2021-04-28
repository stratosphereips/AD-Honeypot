#!/usr/bin/env python3
#Author: Ondrej Lukas - lukasond@fel.cvut.cz
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
from utils import *
#***********************************************************
#                      Models                               
############################################################
""""
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
        hidden = tf.transpose(tf.ones([hidden.shape[-1], 1, 1])*tf.cast(mask, tf.float32), [1, 2, 0])*hidden
        predictions = self.pred(hidden)
        return predictions

class Model2(tf.keras.Model):
    def __init__(self,args):
        super(Model2,self).__init__(name="Model2 - AutoEncoder")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = DAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True, return_Hi=True)
        self.pred = CartesianProductClassifier_2inputs()
        self.h_estimator = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(256, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(args.rnn_dim)])

    def call(self,inputs):
        X,A = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)

        #encdode nodes in existing graph
        hidden_nodes, hidden_G = self.encoder([x_emb, A])
        #apply mask
        hidden_nodes = tf.transpose(tf.ones([hidden_nodes.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden_nodes
        hidden_G = tf.transpose(tf.ones([hidden_G.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden_G
        concat = tf.concat([hidden_G, x_emb],axis=-1)
        estimated_h = self.h_estimator(concat)
        predictions = self.pred([hidden_nodes, estimated_h])
        return predictions, hidden_nodes, estimated_h

class BiDAG_AE(tf.keras.Model):
    def __init__(self,args):
        super(BiDAG_AE,self).__init__(name="BiDAG_AE")
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

class DAG_AE_newX(tf.keras.Model):
    def __init__(self,args):
        super(DAG_AE_newX,self).__init__(name="DAG_AE_newX")
        self.node_embedding = tf.keras.layers.Embedding(input_dim=args.feature_dim, output_dim=args.node_emb_dim,mask_zero=True)
        self.encoder = BidirectionalDAG_RNN(rnn_dim=args.rnn_dim, return_sequences=True)
        #self.size_m = tf.keras.models.Sequential([tf.keras.layers.Dense(32,"relu"), tf.keras.layers.Dense(1,"relu")])
        self.dim_aligner = tf.keras.layers.Dense(args.rnn_dim)
        self.pred = CartesianProductClassifier_2inputs()

    def call(self,inputs):
        X,A,X_hat = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        x_hat_emb = self.node_embedding(X_hat)
        upsampled = self.dim_aligner(x_hat_emb)
        #encdode nodes in existing graph
        hidden= self.encoder([x_emb, A])
        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        #upsampled = tf.transpose(tf.ones([upsampled.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*upsampled
        predictions = self.pred([hidden,upsampled])
        return predictions
"""



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
        self.pred = CartesianProductClassifier_2inputs()
        self._z_dim = args.rnn_dim

    def call(self,inputs):
        X,A = inputs
        #tf.debugging.assert_all_finite(X, "Input X incorrect")
        #tf.debugging.assert_all_finite(A, "Input A incorrect")
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X)
        tf.debugging.assert_all_finite(x_emb, "Embedding incorrect")
        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A])
        #tf.debugging.assert_all_finite(hidden, "hidden incorrect")

        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        #get latent space parameters
        #tf.debugging.assert_all_finite(hidden, "hidden masked incorrect")
        z_mean = self.z_mean(hidden)
        z_log_variance = self.z_log_variance(hidden)
        #tf.debugging.assert_all_finite(z_mean, "z_mean incorrect")
        #tf.debugging.assert_all_finite(z_log_variance, "z_log_variance incorrect")

        #tf.print(z_mean,output_stream="file://test.txt",summarize=-1)
        #tf.print("var:",output_stream="file://test.txt",summarize=-1)
        #tf.print(z_log_variance,tf.int32,output_stream="file://test.txt",summarize=-1)
        #tf.print(hidden.shape,tf.int32,output_stream="file://test.txt",summarize=-1)
        z = tf.random.normal(shape=hidden.shape, mean=z_mean, stddev=tf.math.exp(z_log_variance/2))
        #tf.debugging.assert_all_finite(z, "Sampled z incorrect")
        #tf.print(z.shape,hidden.shape)
        predictions = self.pred([hidden,z])
        return predictions,z_mean,z_log_variance

    def generate(self, inputs):
        X,A,m = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X,training=False)

        tf.debugging.assert_all_finite(x_emb, "Embedding incorrect")
        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A],training=False)
        #tf.debugging.assert_all_finite(hidden, "hidden incorrect")

        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        z_mean = self.z_mean(hidden)
        z_log_variance = self.z_log_variance(hidden)
        #sample z from N~(0,1)
        z = tf.random.normal(shape=[hidden.shape[0],m,hidden.shape[2]],mean=0, stddev=1)
        tf.print("SAMPLED SHAPE:", z.shape)
        tf.print(z,output_stream="file://sampled_strato.txt",summarize=-1)
        predictions = self.pred([hidden,z],training=False)
        tf.print(predictions.shape)
        return tf.transpose(predictions, [1,0]), z

    def get_latent_representaition(self, inputs):
        X,A = inputs
        mask = tf.not_equal(X, 0)
        # Embedd input nodes
        x_emb = self.node_embedding(X,training=False)
        tf.debugging.assert_all_finite(x_emb, "Embedding incorrect")
        #encdode nodes in existing graph
        hidden = self.encoder([x_emb, A],training=False)
        #tf.debugging.assert_all_finite(hidden, "hidden incorrect")

        #apply mask
        hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        return hidden
############################################################
#                   Networks                               # 
############################################################
"""
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
        #Method for computing KL divergence of two normal distributions.
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / (b_sd_squared1e-12)
        return (a_mean - b_mean) ** 2 / ((2 * b_sd_squared)+1e-12) + (ratio - tf.math.log(ratio) - 1) / 2

    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            #self._model.build([X.shape,A.shape])
            #self._model._set_inputs([X,A])
            self._model([X,A])
        with tf.GradientTape() as tape:
            
            #find a mask of the inputX
            #mask = tf.not_equal(X, 0)
            # Embedd input nodes
            #x_emb = self._model.node_embedding(X,training=True)

            #encdode nodes in existing graph
            #hidden = self._model.encoder([x_emb, A])

            #apply mask
            #hidden = tf.transpose(tf.ones([hidden.shape[-1],1,1])*tf.cast(mask,tf.float32),[1,2,0])*hidden
        
            #predictions = self._model.pred(hidden,training=True)
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
            #print(mask_a.shape)
            loss = self.train_batch(X,A, mask_a)


    def train(self,dataset_train,dataset_dev,args):
        for epoch in range(args.epochs):
            self.train_epoch(dataset_train,args)
            metrics_eval = self.evaluate(dataset_dev, args)
            tf.print(f"Epoch {epoch+1}/{args.epochs}: {metrics_eval}")

    def predict_batch(self,X,A,mask_a=None):
        mask = tf.not_equal(X, 0)
        #tf.print(mask.shape)
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

        for i in range(len(batch["X"])):
            tf.print("target:",output_stream="file://output.txt")
            tf.print(tf.cast(batch["A"][i],tf.int32),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted:",output_stream="file://output.txt")
            tf.print(tf.where(predictions[i,:]>=args.edge_threshold,1,0),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted_raw:",output_stream="file://output.txt")
            tf.print(predictions[i,:],summarize=-1,output_stream="file://output.txt")
            tf.print("---------------------------------------",output_stream="file://output.txt")
        return metrics, predictions

    def generate(self, X, A, new_node_types):
        # Embedd input nodes
        x_emb = self._model.node_embedding(X,training=False)

        #encdode nodes in existing graph
        hidden = self._model.encoder(x_emb, A)

        #generated hidden states for new nodes

        #predict
        predictions = self._model.pred(hidden,training=False)
        
        #replace predictions of original nodes with the real values
        return predictions
"""
"""
class ModelType2:
    def __init__(self, args):
        self._model = Model2(args)  
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.95, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._loss = SigmoidFocalCrossEntropy()
        self._loss2 = tf.keras.losses.Huber()
        self._metrics = {"loss":tf.keras.metrics.Mean(),"ROC":tf.keras.metrics.AUC(), "AUC_PR":tf.keras.metrics.AUC(curve="PR"), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self._z_dim = args.rnn_dim
        self.verbose = args.verbose


    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            #self._model.build([X.shape,A.shape])
            #self._model._set_inputs([X,A])
            self._model([X,A])
        with tf.GradientTape() as tape:
            predictions, hidden_nodes, h_estimate = self._model([X,A],training=True)
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a

    
            rec = tf.reduce_mean(self._loss(A, predictions))
            h_loss = tf.reduce_mean(self._loss2(hidden_nodes, h_estimate))
            loss = rec*A.shape[1]*A.shape[1] + h_loss*self._z_dim
    
            #loss = self._loss(A,predictions)


            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, 10.0)
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
            #print(mask_a.shape)
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
        
        #loss = self._loss(A,predictions)
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

        for i in range(len(batch["X"])):
            tf.print("target:",output_stream="file://output.txt")
            tf.print(tf.cast(batch["A"][i],tf.int32),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted:",output_stream="file://output.txt")
            tf.print(tf.where(predictions[i,:]>=args.edge_threshold,1,0),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted_raw:",output_stream="file://output.txt")
            tf.print(predictions[i,:],summarize=-1,output_stream="file://output.txt")
            tf.print("---------------------------------------",output_stream="file://output.txt")
        return metrics, predictions

    def generate(self, X, A, new_node_types):
        # Embedd input nodes
        x_emb = self._model.node_embedding(X,training=False)

        #encdode nodes in existing graph
        hidden = self._model.encoder(x_emb, A)

        #generated hidden states for new nodes

        #predict
        predictions = self._model.pred(hidden,training=False)
        
        #replace predictions of original nodes with the real values
        return predictions
"""
class DAG_RNNVariationalAutoencoder:
    def __init__(self, args):
        self._model = DAG_VAE(args)
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.98, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._reconstruction_loss = SigmoidFocalCrossEntropy()
        self._metrics = {"loss":tf.keras.metrics.Mean(),"ROC":tf.keras.metrics.AUC(), "AUC_PR":tf.keras.metrics.AUC(curve="PR"), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self._z_dim = args.rnn_dim
        self.verbose = args.verbose
        tf.print("USING MODEL DAG-RNN VAE")
    
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

        for i in range(len(batch["X"])):
            tf.print("target:",output_stream="file://output.txt")
            tf.print(tf.cast(batch["A"][i],tf.int32),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted:",output_stream="file://output.txt")
            tf.print(tf.where(predictions[i,:]>=args.edge_threshold,1,0),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted_raw:",output_stream="file://output.txt")
            tf.print(predictions[i,:],summarize=-1,output_stream="file://output.txt")
            tf.print("---------------------------------------",output_stream="file://output.txt")
        return metrics, predictions

    def generate(self, X, A,args,m):
            predictions, z = self._model.generate([X,A,m])
            return predictions, z

    def get_latent_representaition(self,X,A):
        hidden = self._model.get_latent_representaition([X,A])
        return hidden
"""
class DAG_RNNAutoencoderNewX:
    def __init__(self, args):
        self._model = DAG_AE_newX(args)
        #self._model.build(input_shape=[[None,args.feature_dim],[args.batch_size,None,None]])
        #self._model.summary()
        initial_learning_rate = 0.1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.98, staircase=True)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        self._loss = SigmoidFocalCrossEntropy(alpha=0.01)
        self._metrics = {"loss":tf.keras.metrics.Mean(),"ROC":tf.keras.metrics.AUC(), "AUC_PR":tf.keras.metrics.AUC(curve="PR"), "Rec":tf.keras.metrics.Recall(thresholds=args.edge_threshold),"Prec":tf.keras.metrics.Precision(thresholds=args.edge_threshold)}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
        self.verbose = args.verbose

    def train_batch(self,X,A,mask_a=None):
        if not self._model.built:
            self._model([X,A,X])
        with tf.GradientTape() as tape:
            #predictions = self._model.pred(hidden,training=True)
            predictions = self._model([X,A,X],training=True)
            predictions = tf.transpose(predictions,perm=[0,2,1])
            #mask predictions using the mask_a input (set values out of the graph to zero)
            predictions = predictions*mask_a
            loss = self._loss(A, predictions)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            #clip gradients using the global norm
            #gradients = [tf.clip_by_norm(g, 10.0) for g in gradients]
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
            #tf.summary.scalar("train/gradient_norm", gradient_norm)

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
            _ = self.evaluate_batch(X,A, mask_a)

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

        for i in range(len(batch["X"])):
            tf.print("target:",output_stream="file://output.txt")
            tf.print(tf.cast(batch["A"][i],tf.int32),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted:",output_stream="file://output.txt")
            tf.print(tf.where(predictions[i,:]>=args.edge_threshold,1,0),output_stream="file://output.txt",summarize=-1)
            tf.print("predicted_raw:",output_stream="file://output.txt")
            tf.print(predictions[i,:],summarize=-1,output_stream="file://output.txt")
            tf.print("---------------------------------------",output_stream="file://output.txt")
        return metrics, predictions

    def generate(self, X, A, new_node_types):
        # Embedd input nodes
        x_emb = self._model.node_embedding(X,training=False)

        #encdode nodes in existing graph
        hidden = self._model.encoder(x_emb, A)

        #generated hidden states for new nodes

        #predict
        predictions = self._model.pred(hidden,training=False)
        
        #replace predictions of original nodes with the real values
        return predictions

"""
if __name__ == "__main__":
    print(tf.__version__)
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    #global parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    
    #encoder parameters
    parser.add_argument("--feature_dim", default=6, type=int, help="Feature dimension")
    parser.add_argument("--rnn_dim", default=32, type=int, help="Encoder RNN dimension.")
    parser.add_argument("--node_emb_dim", default=6, type=int, help="Graph embedding dimension")
    parser.add_argument("--edge_threshold", default=0.05, type=float, help="Edge limit value during generation")
    parser.add_argument("--verbose", default=False, type=bool, help="Verbose output")
    args = parser.parse_args()
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    #set amount of cores to use
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    #physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    #train_data = "./dag-data/150_nodes/train_150.pickle"
    #dev_data = "./dag-data/150_nodes/dev_150.pickle"

    #train_data = "./dag-data/50_nodes/train_50.pickle"
    #dev_data = "./dag-data/50_nodes/dev_50.pickle"

    train_data = "./dag-data/15_nodes/train_15.pickle"
    dev_data = "./dag-data/15_nodes/dev_15.pickle"

    #train_data = "./dag-data/500_nodes/train_500.pickle"
    #dev_data = "./dag-data/500_nodes/dev_500.pickle"


    #############GRID#######################
    #train_data = "./dag-data/dataset_grid_50.pickle"
    #dev_data = "./dag-data/dataset_grid_50_dev.pickle"

    dataset_train = DAG_Dataset(shuffle_batches=True,max_samples=2000,seed=42)
    dataset_train.read_nx_pickle(train_data)
    dataset_dev = DAG_Dataset(shuffle_batches=True, max_samples=200,seed=42)
    dataset_dev.read_nx_pickle(dev_data)
    
    #network = DAG_RNNAutoencoderNewX(args)
    network = DAG_RNNVariationalAutoencoder(args)
    #network = ModelType2(args)
    #network = DAG_RNNAutoencoder(args)
    
    tf.print("Build complete")
    tf.print(f"TRAIN dataset:{train_data.split('/')[-1]}, size:{dataset_train.size()}")
    tf.print(f"DEV dataset:{dev_data.split('/')[-1]}, size:{dataset_dev.size()}")
    arg_list  = [(k,v) for k, v in sorted(vars(args).items()) if k != 'logdir']
    details = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in arg_list))
    network.train(dataset_train, dataset_dev, args)
    network._model.save_weights(f'./VAE_{details}/my_checkpoint')
    #network._model.load_weights(f'./VAE_{details}/my_checkpoint')
    metrics, predictions = network.test(dataset_dev, args)
    predictions = []
    for batch in dataset_dev.batches(1):
        X = batch["X"]
        A = batch["A"]
        pred, z  = network.generate(X, A, args, 3)
        predictions.append({"X":X, "A":A, "A_hat":pred, "z":z})
    out = {"predictions":predictions, "info":details}
    with open(f'generated_grid50.pickle','wb') as f: pickle.dump(out, f)
    """
    print("TEST:")
    #print(metrics)
    print("Generation")
    
    #for batch in dataset_dev.batches(1):
    #        X = batch["X"]
    #        A = batch["A"]
    #pred, z  = network.generate(X, A, args, 10)

    #tf.print(pred,output_stream="file://generated.txt",summarize=-1)
    
    with open('./real-data/20201210193734_Stratodomain_original/preprocessed_stratotest_domain.pickle','rb') as in_f: real_data = pickle.load(in_f)
    X,A = tf.expand_dims(tf.cast(real_data["X"], tf.float32),axis=0), tf.expand_dims(tf.cast(real_data["A"], tf.float32),axis=0)
    A = tf.transpose(A,[0,2,1])
    pred, z  = network.generate(X, A, args, 20)
    tf.print(pred,output_stream="file://generated_strato.txt",summarize=-1)
    with open('generated_strato_pickle.pickle','wb') as f: pickle.dump(pred, f)
    """