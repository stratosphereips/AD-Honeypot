import tensorflow as tf
import numpy as np
import recurrent_fixed

class DeepSetAggregationLayer(tf.keras.Model):
    #based on https://arxiv.org/abs/1703.06114
    def __init__(self, inner_fn, outer_fn, array_input=True, time_first=False):
        super(DeepSetAggregationLayer, self).__init__(self)
        self.inner_fn = inner_fn
        self.outer_fn = outer_fn
        self.array_input = array_input
        self.time_first = time_first

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
        for i in range(input_array.size()):
            tmp_array = tmp_array.write(i,self.inner_fn(input_array.read(i)))
        aggregated = tf.math.reduce_sum(tmp_array.stack(), axis=0)
        return self.outer_fn(aggregated)

class DynamicRNNEncoder(tf.keras.Model):

    def __init__(self, rnn_dim, agg_hidden_size, z_dim):
        super(DynamicRNNEncoder, self).__init__(self)
        self.cell = recurrent_fixed.GRUCell(rnn_dim)
        self.output_aggregator = DeepSetAggregationLayer(tf.keras.layers.Dense(agg_hidden_size, activation="relu"), tf.keras.layers.Dense(z_dim, activation="relu"),array_input=True)
  
    @tf.function
    def call(self, input_data, conditions):
        # [batch, time, features] -> [time, batch, features]
        input_data = tf.transpose(input_data, [1, 0, 2])
        #[batch]
        conditions = tf.transpose(conditions, [1, 2, 0])
        #outputs = tf.TensorArray(tf.float32, input_data.shape[0] + 1)
        next_state = self.cell.get_initial_state(input_data[0,:,:])
        outputs = tf.TensorArray(tf.float32, 0,dynamic_size=True, element_shape=next_state.shape)
        #next_state = self.cell.get_initial_state(input_data[0,:,:])
        if input_data.shape[0] == None:
            l = 100
        else:
            l = input_data.shape[0]
            for i in range(input_data.shape[0]):
                outputs = outputs.write(i,tf.zeros_like(next_state))
        for i in tf.range(l):
            if i > 0:
                previous_states = outputs.identity()
                previous_states = previous_states.stack()

                mask = conditions[i]
                
                mask = tf.stack([mask for k in range(previous_states.shape[-1])],axis=2)
                #print(mask.shape, previous_states.shape)
                masked = previous_states*mask
                aggregated = tf.math.reduce_sum(masked,axis=0)
                #prepare next 
                next_state = aggregated
            output, state = self.cell(input_data[i], next_state)
            #append to the TensorArray
            outputs = outputs.write(i, output)

        if self.output_aggregator:
            return self.output_aggregator(outputs)
        else:
          return tf.transpose(outputs.stack(), [1, 0, 2]) #return the stacked array  

class MLP(tf.keras.Model):

    def __init__(self, attributes):
        super(MLP, self).__init__(self)
        self.l= []
        for i in range(len(attributes)):
            self.l.append(tf.keras.layers.Dense(attributes[i][0],activation=attributes[i][1]))
    @tf.function
    def call(self, inputs):
        tmp  = inputs
        for layer in self.l:
            tmp = layer(tmp)
        return tmp

class DynamicRNNDecoder(tf.keras.Model):


    def __init__(self, rnn_dim,z_dim, feature_dim, edge_treshold=0.5, modul_hidden_size=128,agg_hidden_size=64):
        super(DynamicRNNDecoder, self).__init__(self)
        self.feature_dim = feature_dim
        self.edge_treshold = edge_treshold
        self.rnn_cell = recurrent_fixed.GRUCell(rnn_dim)

        self.N_module = out_mean = MLP([(modul_hidden_size,"relu"),(1,"relu")])

        self.feature_module = MLP([(modul_hidden_size,"relu"),(feature_dim,"softmax")])
        self.edge_module = MLP([(modul_hidden_size,"relu"),(1,"sigmoid")])
        self.init_module = MLP([(modul_hidden_size,"relu"),(rnn_dim,"relu")])
        self.graph_aggregator = DeepSetAggregationLayer(tf.keras.layers.Dense(agg_hidden_size,activation="relu"),tf.keras.layers.Dense(z_dim,activation="relu"))
        self.node_aggregator = DeepSetAggregationLayer(tf.keras.layers.Dense(agg_hidden_size,activation="relu"),tf.keras.layers.Dense(self.rnn_cell.units,activation="relu"))

    @tf.function
    def call(self, in_encoding):

        #in_encoding shape [1,encoding_dim] - [batch_size, embedding_size]
        #target_nodes shape [1,1] - [batch_size, 1]

        #estimate number of nodes in the graph usion Poisson distribution and given lambdas
        lambdas = self.N_module(in_encoding)
        
        N = tf.squeeze(tf.random.poisson([1],lambdas, dtype=tf.dtypes.float32))
        N = tf.cast(N,tf.int32)
        self._test = N
        max_nodes = tf.reduce_max(N)
        #test = [x for x in N]
        if max_nodes <= 0:
            max_nodes =1
        batch_size = lambdas.shape[0] if lambdas.shape[0] else 1

        #self._feature_mask = None
        h_nodes = tf.TensorArray(dtype=tf.float32,size=max_nodes,clear_after_read = False).unstack(tf.zeros([max_nodes, batch_size, self.rnn_cell.units]))
        f_nodes = tf.TensorArray(dtype=tf.float32,size=max_nodes,clear_after_read = False).unstack(tf.zeros([max_nodes, batch_size, self.feature_dim]))
        e_nodes = tf.TensorArray(dtype=tf.float32,size=max_nodes,clear_after_read = False).unstack(tf.zeros([max_nodes, max_nodes, batch_size]))

        #initialize hidden state of first node and graph encoding
        h_nodes = h_nodes.write(0,self.init_module(in_encoding))
        H_graph = in_encoding

        #estimate features of the first node
        f_nodes = f_nodes.write(0,self.feature_module(h_nodes.read(0)))
        
        
        add_edges = tf.TensorArray(dtype=tf.float32,size=max_nodes,clear_after_read = False,element_shape=e_nodes.read(0).shape)
        for i in range(max_nodes):
            add_adges = add_edges.write(i,tf.zeros_like(e_nodes.read(0)))
        for i in tf.range(1,max_nodes):
            for j in tf.range(i):
                #join values of hidden state of Node j([batch,rnn_dim]) and H_graph[batch, encoding_dim] -> [batch,rnn_dim + encoding_dim]
                merged = tf.concat([H_graph, h_nodes.read(j)],axis=1) 
                #estimate edges
                add_adge = self.edge_module(merged) #shape [batch,1]
                add_edges = add_edges.write(j,tf.transpose(add_adge))
            e_nodes = e_nodes.write(i,add_edges.concat())
            #cleanup the temporal array
            for l in range(max_nodes):
                add_adges = add_edges.write(l,tf.zeros_like(e_nodes.read(0)))

            #make a copy of hidden states and create tensor
            previous_states = h_nodes.identity()
            previous_states = previous_states.stack()
            
            #aggregate ancestors
            mask_tmp = e_nodes.read(i) #shape [N,batch]
            mask = tf.stack([mask_tmp for k in range(previous_states.shape[-1])],axis=2)
            mask = tf.where(mask >= self.edge_treshold,1.0,0.0)
            masked = previous_states*mask
            ancestors = tf.math.reduce_sum(masked,axis=0)

            #find sinknodes
            children_view = e_nodes.identity().stack()

            has_children_mask = tf.math.reduce_sum(children_view,axis=0)
            sink_mask = tf.where(has_children_mask < self.edge_treshold, 1.0,0.0)
            sink_mask = tf.stack([sink_mask for k in range(previous_states.shape[-1])],axis=2)
            sink_masked = previous_states*sink_mask
            sink_nodes = tf.math.reduce_sum(sink_masked,axis=0)

            #get current node encoding from RNN cell
            next_node_h = self.rnn_cell(sink_nodes,ancestors)[0]
            h_nodes = h_nodes.write(i,next_node_h)
            #find features
            f_nodes = f_nodes.write(i,self.feature_module(next_node_h))
            
            #update graph encoding
            H_graph = self.graph_aggregator(h_nodes.identity())
        
        return lambdas,tf.transpose(f_nodes.stack(), [1, 0, 2]), tf.transpose(e_nodes.stack(), [2, 0, 1]), N