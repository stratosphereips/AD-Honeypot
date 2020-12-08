import tensorflow as tf
import numpy as np
import time
class DAG_RNN(tf.keras.layers.Layer):
    def __init__(self, rnn_dim, return_sequences=True, go_backwards=False, return_Hi=False):
        super(DAG_RNN, self).__init__(self)
        self.cell = tf.keras.layers.GRUCell(rnn_dim)
        self.return_sequences = return_sequences
        self.reverse = go_backwards
        self.return_Hi = return_Hi
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs_data,conditions = inputs
        input_data = tf.transpose(inputs_data, [1, 0, 2])

        #[batch,node,parents] -> [node, parents,batch]
        if self.reverse:
            conditions = tf.transpose(conditions, [2,1,0])
        else:
            conditions = tf.transpose(conditions, [1,2,0])
        #prepare next state for RNN
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

                next_state = tf.math.reduce_sum(masked,axis=0)
                output, state = self.cell(input_data[i], next_state)
                
                #append to the TensorArray
                outputs = outputs.write(i, output)
                if self.return_Hi:
                    H_g = H_g.write(i, tf.reduce_sum(outputs.identity().stack(),axis=0))
        else:
            for i in tf.range(len(input_data)):
                previous_states = outputs.identity().stack() #[i,batch,rnn_dim]
                mask_p = tf.transpose(tf.ones([previous_states.shape[-1],1,1])*conditions[i],[1,2,0])
                masked = previous_states*mask_p
                next_state = tf.math.reduce_sum(masked,axis=0)
                output, state = self.cell(input_data[i], next_state)
                #append to the TensorArray
                outputs = outputs.write(i, output)
                if self.return_Hi:
                    H_g = H_g.write(i, tf.reduce_sum(outputs.identity().stack(),axis=0))
        outputs = outputs.identity().stack() #[max_n, batch, rnn_dim]
        if self.return_sequences:
            if self.return_Hi:
                return tf.transpose(outputs,[1,0,2]), tf.transpose(H_g.stack(),[1,0,2])
            else:
                return tf.transpose(outputs,[1,0,2])
        else:
            combined = tf.math.reduce_sum(outputs,axis=0)
            return combined
    
    def compute_output_shape(self,input_shape):
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

class CartesianMLPDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super(CartesianMLPDecoder, self).__init__(self)
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

class EdgePredictor(tf.keras.layers.Layer):
        def __init__(self):
            super(EdgePredictor, self).__init__(self)
            self._model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, "relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(32, "relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, "relu"),
                tf.keras.layers.Dense(1, "sigmoid")],
                name="d_edge_module")

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
            out = self._model(out)
            return tf.squeeze(out)

if __name__ == '__main__':
    X = np.array([[1,2,3,0]],dtype=np.float32)
    A = np.array([[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,0]]],dtype=np.float32)
    targets = [[0,1,0,0],[0,0,0,1]]
    cp  = CartesianMLPDecoder()
    pred = EdgePredictor()
    x = tf.constant([[[0,0,0],[2,2,2],[3,3,3]]],dtype=tf.float32)
    y = tf.constant([[[5,5,5,5]]],dtype=tf.float32)
    #print(cp([x,y]))
    n =tf.constant([[[4,0,0,0],[2,0,0,0]]],dtype=tf.float32)
    print(pred([x,n]))