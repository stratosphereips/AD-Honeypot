import tensorflow as tf
import numpy as np


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
            input_array = tf.TensorArray(dtype=tf.float32,size=input_data.shape[0]).unstack(input_data)
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

class DynamicRNNEncoder(tf.keras.Model):

    def __init__(self, rnn_dim, return_sequences=True, go_backwards=False):
        super(DynamicRNNEncoder, self).__init__(self)
        self.cell = tf.keras.layers.GRUCell(rnn_dim)
        self.return_sequences = return_sequences
        self.reverse = go_backwards

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, conditions):
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
        if self.reverse:
            for i in tf.range(len(input_data)-1,-1,-1):
                previous_states = outputs.identity().stack() #[i,batch,rnn_dim]

                #mask_p2 = tf.stack([conditions[i]  for k in range(previous_states.shape[-1])],axis=2)
                mask_p = tf.transpose(tf.ones([previous_states.shape[-1],1,1])*conditions[i],[1,2,0])

                masked = previous_states*mask_p
                next_state = tf.math.reduce_sum(masked,axis=0)
                output, state = self.cell(input_data[i], next_state)
                #append to the TensorArray
                outputs = outputs.write(i, output)
        else:
            for i in tf.range(len(input_data)):
                previous_states = outputs.identity().stack() #[i,batch,rnn_dim]
                #mask_p = tf.stack([conditions[i]  for k in range(previous_states.shape[-1])],axis=2)
                mask_p = tf.transpose(tf.ones([previous_states.shape[-1],1,1])*conditions[i],[1,2,0])
                masked = previous_states*mask_p
                next_state = tf.math.reduce_sum(masked,axis=0)
                output, state = self.cell(input_data[i], next_state)
                #append to the TensorArray
                outputs = outputs.write(i, output)
        outputs = outputs.identity().stack() #[max_n, batch, rnn_dim]
        if self.return_sequences:
            return tf.transpose(outputs,[1,0,2])
        else:
            added = tf.math.reduce_sum(outputs,axis=0)
            return added
"""
class DynamicRNNDecoder(tf.keras.Model):


    def __init__(self, rnn_dim, z_dim, feature_dim, edge_threshold=0.5, modul_hidden_size=128):
        super(DynamicRNNDecoder, self).__init__(self)
        self.feature_dim = feature_dim
        self.edge_threshold = edge_threshold
        self.rnn_cell = tf.keras.layersGRUCell(rnn_dim)
        #node features module
        self.feature_module = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512,"tanh"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256,"tanh"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(feature_dim,"softmax")],
            name="d_feature_module")
        #RNN intitial state module
        self.init_module = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(rnn_dim, None)],
            name="d_init_module")
        #edge estimation module
        self.edge_module = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, "relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1,None)],
            name="d_edge_module"
        )
    

    @tf.function
    def call(self, in_encoding, size,gold_A,training=None):
        
        N = tf.cast(tf.squeeze(size),tf.int32)
        max_nodes = tf.cast(tf.reduce_max(N),tf.int32)

        batch_size = in_encoding.get_shape().as_list()[0]
        batch_size = batch_size if batch_size else 1
        gold_A =  tf.transpose(gold_A, [1,2,0])
       
        #initialize arrays for storing prictions
        h_nodes = tf.TensorArray(dtype=tf.float32,size=0,clear_after_read = False, dynamic_size=True, element_shape=[batch_size, self.rnn_cell.units])
        f_nodes = tf.TensorArray(dtype=tf.float32,size=0,clear_after_read = False, dynamic_size=True, element_shape=[batch_size, self.feature_dim])
        
        e_nodes = tf.TensorArray(dtype=tf.float32,size=1,clear_after_read = False, dynamic_size=True).unstack(tf.zeros([max_nodes, max_nodes, batch_size]))
        e_nodes_binary = tf.TensorArray(dtype=tf.float32,size=1,clear_after_read = False, dynamic_size=True).unstack(tf.zeros([max_nodes, max_nodes, batch_size]))
        
        #initialize hidden state of first node and graph encoding
        h_nodes = h_nodes.write(0,self.init_module(in_encoding))
        H_graph = in_encoding

        #estimate features of the first node
        f_nodes = f_nodes.write(0,self.feature_module(h_nodes.read(0)))

    
        idx = 1
        for i in tf.range(1,max_nodes):
            #create tmporal array for edges to node i
            add_edges = tf.TensorArray(dtype=tf.float32,size=max_nodes, clear_after_read = False).unstack(tf.zeros([max_nodes, 1, batch_size]))   
            for j in tf.range(i):
                #join values of hidden state of Node j([batch,rnn_dim]) and H_graph[batch, encoding_dim] -> [batch,rnn_dim + encoding_dim]
                merged = tf.concat([H_graph, h_nodes.read(j)],axis=1) 
                #estimate edge from node j to node i
                add_edge = self.edge_module(merged) #shape [batch,1]
                #write result in tmp array
                add_edges = add_edges.write(j,tf.transpose(add_edge))    
            
            #covert to binary values based on self.edge_threshold
            raw = add_edges.concat()
            e_nodes = e_nodes.write(i,raw)
            binary = tf.where(raw >= self.edge_threshold, 1.0, 0.0)
            
            #write in the edge array
            e_nodes_binary = e_nodes.write(i, binary)
            
            #make a copy of hidden states and create tensor
            previous_states = h_nodes.identity()
            previous_states = previous_states.stack()

            
            mask_tmp = e_nodes_binary.read(i)[:idx,:] #shape [i:idx, batch]
            
            mask = tf.stack([mask_tmp for k in range(previous_states.shape[-1])],axis=2)
    
            masked = previous_states*mask
    
            ancestors = tf.math.reduce_sum(masked,axis=0)
            
            
            #find sinknodes
            children_view = e_nodes_binary.identity().stack()
            
            if training:
                 children_view = gold_A[:idx,:idx,:]
            else:
                children_view = children_view[:idx,:idx,:]
        
            has_children_mask = tf.math.reduce_sum(children_view,axis=0)
            
            sink_mask = tf.where(has_children_mask == 0, 1.0,0.0)
           
            sink_mask = tf.stack([sink_mask for k in range(previous_states.shape[-1])],axis=2)
            
            sink_masked = previous_states*sink_mask
            
            sink_nodes = tf.math.reduce_sum(sink_masked,axis=0)

            #get current node encoding from RNN cell
            next_node_h = self.rnn_cell(sink_nodes,ancestors)[0]
            h_nodes = h_nodes.write(i,next_node_h)
            
            #find features
            f_nodes = f_nodes.write(i, self.feature_module(tf.concat(next_node_h,axis=1)))
            
            #update graph encoding
            H_graph = tf.math.reduce_sum(h_nodes.identity().stack(),axis=0)
            idx += 1
        return tf.transpose(f_nodes.stack(), [1, 0, 2]), tf.transpose(e_nodes.stack(),[2,0,1]) #tf.transpose(gold_A,[2,0,1])
"""
class EdgePredictor(tf.keras.Model):


    def __init__(self):
        super(EdgePredictor, self).__init__(self)
        self.edge_module = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, "relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(1, "sigmoid")],
            name="d_edge_module")


    @tf.function(experimental_relax_shapes=True)
    def call(self, nodes_hidden, new_nodes):
        new_embedded = tf.squeeze(new_nodes,axis=1)
        
        inputs = tf.transpose(nodes_hidden,[1,0,2])
        tmp = tf.ones([len(inputs),1,1])*new_embedded

        con = tf.concat([inputs, tmp],axis=2)
        edges = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        for i in tf.range(len(inputs)):
            edges = edges.write(i,self.edge_module(con[i]))

        ret = tf.transpose(edges.stack(),[1,0,2])
        return tf.squeeze(ret)


if __name__ == '__main__':
    X = np.array([[1,2,3,0],[1,3,3,2]],dtype=np.float32)
    A = np.array([[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,0]],[[0,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,0]]],dtype=np.float32)
    N = np.array([[2],[4]])
    targets = [[0,1,0,0],[0,0,0,1]]

    l = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy()
    node_embedding = tf.keras.layers.Embedding(input_dim=8, output_dim=5)
    forw = DynamicRNNEncoder(rnn_dim=4,return_sequences=True,go_backwards=False)
    backw = DynamicRNNEncoder(rnn_dim=4,return_sequences=True, go_backwards=True)
    pred = EdgePredictor()
    
    

    mask = tf.not_equal(X, 0)
    tf.print(mask)
    print("----------------------")
    emb = node_embedding(X)
    tf.print(emb.shape)
    tf.print("----------------")
    f = forw(emb,A)
    tf.print(f.shape)
    tf.print("----------------")
    b = backw(emb,A)
    tf.print(b.shape)
    tf.print("----------------")
    merged = tf.keras.layers.Add()([f,b])
    predicted = pred(merged, node_embedding(N))
    tf.print(predicted)
    tf.print(l(targets,predicted))

    masked = tf.where(mask,predicted,0.0)
    tf.print(masked)
    tf.print(l(targets,masked))
    tf.print(metric(targets,predicted))
    tf.print(metric(targets,masked))

