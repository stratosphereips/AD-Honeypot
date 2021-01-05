#!/usr/bin/env python3
#Author: Ondrej Lukas - lukasond@fel.cvut.cz
import tensorflow as tf
import numpy as np
import time
#MODELS for testing
"""
    class DeepSetAggregationLayer(tf.keras.Model):
        #based on https://arxiv.org/abs/1703.06114
        #TODO
        def __init__(self, inner_fn, outer_fn, array_input=True, time_first=False, name=None):
            super(DeepSetAggregationLayer, self).__init__(self)
            self.inner_fn = inner_fn
            self.outer_fn = outer_fn
            self.array_input = array_input
            self.time_first = time_first
            if name:
                self._name = str(name)

        @tf.function(experimental_relax_shapes=True)
        def call(self, input_data):
            if not self.array_input:
                if not self.time_first:
                    #[batch,time,features] -> [time,batch,features]
                    input_data = tf.transpose(input_data,[1,0,2])
                #tf.print(input_data.shape)
                input_array = tf.TensorArray(dtype=tf.float32,size=0).unstack(input_data)
            else:
                input_array = input_data
            tmp_array = tf.TensorArray(dtype=tf.float32,size=input_array.size())
            input_array = input_data
            for i in tf.range(input_array.size()):
                tmp = self.inner_fn(input_array.read(i))
                tmp_array = tmp_array.write(i,tmp)
            aggregated = tf.math.reduce_sum(tmp_array.stack(), axis=0)
            #tf.print(aggregated)
            ret = self.outer_fn(aggregated)
            #tf.print(f"DS{self._name}:",ret)
            return ret

    class DeepSetAggLayer(tf.keras.Model):
        #based on https://arxiv.org/abs/1703.06114
        def __init__(self, output_dim, array_input=False, time_first=True, name=None):
            super(DeepSetAggLayer, self).__init__(self)
            self.i1 = tf.keras.layers.Dense(256, "relu")
            self.i3 = tf.keras.layers.Dense(64, "relu")
            self.outer_fn = tf.keras.layers.Dense(output_dim, None)
            self.output_dim = output_dim
            self.array_input = array_input
            self.time_first = time_first

        @tf.function(experimental_relax_shapes=True)
        def call(self, input_data):
            input_array = tf.TensorArray(dtype=tf.float32,size=1).unstack(input_data)
            tmp_array = tf.TensorArray(dtype=tf.float32,size=input_array.size())
            for i in tf.range(input_array.size()):
                x = self.i1(input_array.read(i))
                tmp = self.i3(x)
                tmp_array = tmp_array.write(i,tmp)
            aggregated = tf.math.reduce_sum(tmp_array.stack(), axis=0)
            ret = self.outer_fn(aggregated)
            return ret

    class EdgePredictor(tf.keras.Model):


        def __init__(self):
            super(EdgePredictor, self).__init__(self)
            self.edge_module = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, "relu"),
                tf.keras.layers.Dense(64, "relu"),
                tf.keras.layers.Dense(1, "sigmoid")],
                name="d_edge_module")


        @tf.function(experimental_relax_shapes=True)
        def call(self, nodes_hidden, new_nodes,training=True):
            nodes_hidden = tf.transpose(nodes_hidden,[1,0,2])
            new_nodes = tf.transpose(new_nodes,[1,0,2])
            #tmp = tf.ones([len(inputs),1,1])*new_embedded

            #con = tf.concat([inputs, tmp],axis=2)
            edges = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
            for i in tf.range(len(nodes_hidden)):
                #tf.print(nodes_hidden[i].shape,new_nodes[0].shape)
                con = tf.concat([nodes_hidden[i],new_nodes[0]],axis=-1)
                edges = edges.write(i,self.edge_module(con, training))

            ret = tf.transpose(edges.stack(),[1,0,2])
            return tf.squeeze(ret)

    class NodeClassifier(tf.keras.Model):


        def __init__(self, n_clases):
            super(NodeClassifier, self).__init__(self)
            self.classifier = tf.keras.models.Sequential([
                tf.keras.layers.Dense(4, "relu"),
                tf.keras.layers.Dense(n_clases, "softmax")],
                name="node_classifier")

        @tf.function(experimental_relax_shapes=True)
        def call(self, nodes_hidden,training=True):
            #new_embedded = tf.squeeze(new_nodes,axis=1)
        
            inputs = tf.transpose(nodes_hidden,[1,0,2])
            #H_g = tf.math.reduce_max(inputs,axis=0)
            #tmp = tf.ones([len(inputs),1,1])*new_embedded

            #con = tf.concat([inputs, tmp],axis=2)
            con = inputs
            out = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
            for i in tf.range(len(inputs)):
               # tmp = tf.concat([con[i],H_g],axis=1)
                out = out.write(i,self.classifier(con[i], training=training))

            ret = tf.transpose(out.stack(),[1,0,2])
            return tf.squeeze(ret)

    class Predictor(tf.keras.layers.Layer):
        def __init__(self):
            super(Predictor, self).__init__(self)
            self._model = [tf.keras.layers.Dense(64,"relu"),
                tf.keras.layers.Dense(64,"relu"),
                tf.keras.layers.Dense(32,"relu"),
                tf.keras.layers.Dense(1, "sigmoid")]
        


        @tf.function(experimental_relax_shapes=True)
        def call(self, inputs):

            def _cartesian_product(a,b):
                a, b = a[ None, :, None ], b[ :, None, None ]
                return tf.concat( [ a + tf.zeros_like( b ),tf.zeros_like( a ) + b,  ], axis = 2 )

            def _get_combinations(inputs):
                batch_size = inputs.shape[0]
                num_nodes = inputs.shape[1]
                hidden_shape = inputs.shape[2]

                cp = _cartesian_product(tf.transpose(inputs,[1,0,2]), tf.transpose(inputs,[1,0,2]))
                return tf.reshape(tf.transpose(cp,[3,0,1,2,4]),[inputs.shape[0],inputs.shape[1],inputs.shape[1],2*inputs.shape[2]])

            t1 = time.time()
            nodes_hidden = tf.transpose(inputs,[1,0,2])
            new_A = tf.TensorArray(dtype=tf.float32,size=len(nodes_hidden))
            for i in tf.range(len(nodes_hidden)):
                A_i = tf.TensorArray(dtype=tf.float32, size=len(nodes_hidden))
                for j in tf.range(len(nodes_hidden)):
                    con = tf.concat([nodes_hidden[j], nodes_hidden[i]],axis=-1)
                    #if i == j:
                    #    A_i = A_i.write(j, tf.zeros_like(con))
                    #else:
                    A_i = A_i.write(j, con)
                one_line = A_i.stack()#tf.squeeze(A_i.stack(),axis=-1)
                new_A = new_A.write(i, one_line)
            ret = tf.transpose(new_A.stack(),[2,0,1,3])
            t2 = time.time()
            #tf.print("---------------------")
            combined = _get_combinations(inputs)
            t3 = time.time()
            tf.print(f"TA:{t2-t1}, CP:{t3-t2}")
            #tf.print("combined")
            #tf.print(combined[:,0,0,:], summarize=-1)
            #tf.print(combined[:,0,1,:], summarize=-1)
            #tf.print(combined[:,0,2,:], summarize=-1)
            #tf.print(combined[:,0,3,:], summarize=-1)
            #tf.print(combined[:,1,0,:], summarize=-1)
            #tf.print(combined[:,1,1,:], summarize=-1)
            #tf.print(combined[:,1,2,:], summarize=-1)
            #tf.print(combined[:,1,3,:], summarize=-1)
            #tf.print("Equal?:",tf.math.equal(ret,combined),summarize=-1)
            for l in self._model:
                ret = l(ret)
            return tf.squeeze(ret)

######################################################################################
######################################################################################
######################################################################################
#WORKING MODELS
"""

"""
Recurent layer for sequential processing Directed Acyclic Graphs. 
Inputs:
    X = Matrix of node features [batch_size, max_nodes, node_space_dim]
    A = Matrix of predeccesors [batch_size, max_nodes, maxnodes]
    Both matrices have to be topologicaly ordered
"""
class DAG_RNN(tf.keras.layers.Layer):
    def __init__(self, rnn_dim, return_sequences=True, go_backwards=False, return_Hi=False):
        super(DAG_RNN, self).__init__(self)
        self.cell = tf.keras.layers.GRUCell(rnn_dim)
        self.return_sequences = return_sequences
        self.reverse = go_backwards
        self.return_Hi = return_Hi
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs,conditions = inputs
        #tf.debugging.assert_all_finite(inputs, "DAG-RNN layer input X all finite check failed!")
        #tf.debugging.assert_all_finite(conditions, "DAG-RNN layer input A all finite check failed!")
        input_data = inputs
        # [batch, time, features] -> [time, batch, features]
        input_data = tf.transpose(input_data, [1, 0, 2])

        #[batch,node,parents] -> [node, parents,batch]
        if self.reverse:
            conditions = tf.transpose(conditions, [2,1,0])
        else:
            conditions = tf.transpose(conditions, [1,2,0])
        
        next_state = self.cell.get_initial_state(input_data[0])

        outputs = tf.TensorArray(dtype=tf.float32, size=len(input_data)).unstack(tf.zeros([len(input_data), len(inputs), self.cell.units]))
        H_g = tf.zeros([1])
        if self.return_Hi:
            H_g = tf.TensorArray(dtype=tf.float32, size=len(input_data)).unstack(tf.zeros([len(input_data), len(inputs), self.cell.units]))
        if self.reverse:
            for i in tf.range(len(input_data)-1,-1,-1):
                previous_states = outputs.identity().stack() #[i,batch,rnn_dim]
                mask_p = tf.transpose(tf.ones([previous_states.shape[-1],1,1])*conditions[i],[1,2,0])
                masked = previous_states*mask_p

                next_state = tf.math.reduce_mean(masked,axis=0)
                #next_state = self.agg(masked)

                output, _ = self.cell(input_data[i], next_state)
                #append to the TensorArray
                outputs = outputs.write(i, output)
                if self.return_Hi:
                    H_g = H_g.write(i, tf.reduce_sum(outputs.identity().stack(),axis=0))
        else:
            for i in tf.range(len(input_data)):
                previous_states = outputs.identity().stack() #[i,batch,rnn_dim]
                mask_p = tf.transpose(tf.ones([previous_states.shape[-1],1,1])*conditions[i],[1,2,0])
                masked = previous_states*mask_p
                next_state = tf.math.reduce_mean(masked,axis=0)
                #next_state = self.agg(masked)
                #tf.print(t_next_state.shape, next_state.shape)
                output, _ = self.cell(input_data[i], next_state)
                #append to the TensorArray
                outputs = outputs.write(i, output)
                if self.return_Hi:
                    H_g = H_g.write(i, tf.reduce_sum(outputs.identity().stack(),axis=0))
        outputs = outputs.identity().stack() #[max_n, batch, rnn_dim]
        #tf.debugging.assert_all_finite(outputs, "DAG-RNN layer output all finite check failed!")
        if self.return_sequences:
            if self.return_Hi:
                return tf.transpose(outputs,[1,0,2]),tf.transpose(H_g.stack(),[1,0,2])
            else:
                return tf.transpose(outputs,[1,0,2])
        else:
            combined = tf.math.reduce_sum(outputs,axis=0)
            return combined
    
    def compute_output_shape(self,input_shape):
        tf.print("COS:", input_shape)
        return tf.TensorShape(input_shape)
"""
    Bidirectional version of DAG-RNN Layer. Both RNN cells have same number of units and type
"""
class BidirectionalDAG_RNN(tf.keras.layers.Layer):
    def __init__(self, rnn_dim, return_sequences=True, merge_mode="SUM"):
        super(BidirectionalDAG_RNN, self).__init__(self)
        self.forward = DAG_RNN(rnn_dim, return_sequences, go_backwards=False)
        self.backward = DAG_RNN(rnn_dim, return_sequences, go_backwards=True)
        self.merge_mode = merge_mode
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        y_f = self.forward(inputs)
        y_b = self.backward(inputs)
        if self.merge_mode == "SUM":
            return tf.keras.layers.Add()([y_f,y_b])
        elif self.merge_mode == "CONCAT":
            return tf.keras.layers.Concatenate()([y_f,y_b])
        else:
            return [y_f, y_b]

class NestedDAG_RNN(tf.keras.layers.Layer):
    def __init__(self,rnn_dim,depth, return_sequences=True):
        super(NestedDAG_RNN, self).__init__(self)
        self.l = [DAG_RNN(rnn_dim=rnn_dim, return_sequences=True) for i in range(depth)]
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        for i in range(0, len(self.l)):
            X = self.l[i]([X,A])
            #X = tf.keras.layers.Add()([X,TMP])
        return X

class CartesianProductClassifier(tf.keras.layers.Layer):
    def __init__(self):
        super(CartesianProductClassifier, self).__init__(self)
        self._model = [
            tf.keras.layers.Dense(64,"relu"),
            tf.keras.layers.Dense(64,"relu"),
            tf.keras.layers.Dense(32,"relu"),
            tf.keras.layers.Dense(1, "sigmoid")]
    

    def call(self, inputs):
        @tf.function
        def _cartesian_product(a,b):
            a, b = a[ None, :, None ], b[ :, None, None ]
            return tf.concat(values=[ a + tf.zeros_like( b ),tf.zeros_like( a ) + b,  ],axis= 2 )
        @tf.function
        def _get_combinations(inputs):
            cp = tf.transpose(_cartesian_product(tf.transpose(inputs,[1,0,2]), tf.transpose(inputs,[1,0,2])),[3,0,1,2,4])
            ret = tf.reshape(cp, [inputs.shape[0],inputs.shape[1],inputs.shape[1],2*inputs.shape[2]])
            return ret

        out = _get_combinations(inputs)
        for l in self._model:
            out = l(out)
        return tf.squeeze(out)

class CartesianProductClassifier_2inputs(tf.keras.layers.Layer):
    def __init__(self):
        super(CartesianProductClassifier_2inputs, self).__init__(self)
        self._model = [
            tf.keras.layers.Dense(64,"relu"),
            tf.keras.layers.Dense(64,"relu"),
            tf.keras.layers.Dense(32,"relu"),
            tf.keras.layers.Dense(1, "sigmoid")]
    
    

    def call(self, inputs):
        @tf.function
        def _cartesian_product(a,b):
            a, b = a[ None, :, None ], b[ :, None, None ]
            return tf.concat( [ a + tf.zeros_like( b ),tf.zeros_like( a ) + b,  ], axis = 2 )
        
        @tf.function
        def _get_combinations(inputs):
            x,y = inputs
            cp = tf.transpose(_cartesian_product(tf.transpose(x,[1,0,2]), tf.transpose(y,[1,0,2])),[3,0,1,2,4])
            ret = tf.reshape(cp, [x.shape[0],x.shape[1],y.shape[1],x.shape[2] + y.shape[2]])
            return ret

        out = _get_combinations(inputs)
        for l in self._model:
            out = l(out)
        return tf.squeeze(out)


if __name__ == '__main__':
    X = np.array([[1,2,3,0]],dtype=np.float32)
    A = np.array([[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,0]]],dtype=np.float32)
    targets = [[0,1,0,0],[0,0,0,1]]
    cp  = CartesianProductClassifier_2inputs()
    x = tf.constant([[[1,1,1],[2,2,2],[3,3,3]]],dtype=tf.float32)
    y = tf.constant([[[5,5,5,5]]],dtype=tf.float32)
    print(cp([x,y]))
