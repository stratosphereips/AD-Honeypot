import tensorflow as tf
import numpy as np
import recurrent_fixed

class DeepSetAggregationLayer(tf.keras.Model):
	def __init__(self,output_inner_fn, output_outer_fn, array_input=False):
		super(DeepSetAggregationLaye, self).__init__(self)
		self.output_inner_fn = output_inner_fn
		self.output_outer_fn = output_outer_fn
		self.array_input = array_input
  
	@tf.function
	def call(self, input_data,time_first=False):
		if not self.array_input:
			if not time_first:
				#[batch,time,features] -> [time,batch,features]
				input_data = tf.transpose(input_data,[1,0,2])
			input_array = tf.TensorArray(dtype=tf.float32,size=input_data.shape[0]).unstack(input_data)
		else:
			input_array = input_data
		tmp_array = tf.TensorArray(dtype=tf.float32,size=input_data.shape[0])
		for i in range(input_array.size()):
			tmp_array.write(i,self.output_inner_fn(input_array.read(i)))
		aggregated = tf.math.reduce_sum(tmp_array.stack(), axis=0)
		return self.output_outer_fn(aggregated)

class DynamicRNNEncoder(tf.keras.Model):

	def __init__(self, rnn_cell,output_inner_fn=None, output_outer_fn=None):
		super(DynamicRNNEncoder, self).__init__(self)
		self.cell = rnn_cell
		self.output_inner_fn = output_inner_fn
		self.output_outer_fn = output_outer_fn
  
	@tf.function
	def call(self, input_data, conditions):
		# [batch, time, features] -> [time, batch, features]
		input_data = tf.transpose(input_data, [1, 0, 2])
		#[batch]
		conditions = tf.transpose(conditions, [1, 2, 0])
		#outputs = tf.TensorArray(tf.float32, input_data.shape[0] + 1)
		next_state = self.cell.get_initial_state(input_data[0,:,:])
		outputs = tf.TensorArray(tf.float32, 0,dynamic_size=True, element_shape=next_state.shape)
		next_state = self.cell.get_initial_state(input_data[0,:,:])
		if input_data.shape[0] == None:
			l = 100
		else:
			l = input_data.shape[0]
			for i in range(input_data.shape[0]):
				outputs = outputs.write(i,next_state)
		for i in tf.range(l):
			if i > 0:
				previous_states = outputs.identity()
				previous_states = previous_states.stack()

				mask = conditions[i]
				#print(mask.shape)
				mask = tf.stack([mask for k in range(previous_states.shape[-1])],axis=2)
				#print(mask.shape, previous_states.shape)
				masked = previous_states*mask
				aggregated = tf.math.reduce_sum(masked,axis=0)
				#prepare next 
				next_state = aggregated
			output, state = self.cell(input_data[i], next_state)
			#append to the TensorArray
			outputs = outputs.write(i, output)

		if self.output_inner_fn: #process the individual node states and aggregate the results 
			tmp = outputs.identity()
			out = tf.TensorArray(tf.float32, tmp.size())
			for i in range(tmp.size()):
				out = out.write(i,self.output_inner_fn(tmp.read(i)))
			aggregated = tf.math.reduce_sum(out.stack(), axis=0)
			if self.output_outer_fn:
				return self.output_outer_fn(aggregated)
			else:
				return aggregated
		else:
			return tf.transpose(outputs.stack(), [1, 0, 2]) #return the stacked array (2nd dim is not fixed!)


class DynamicRNNDecoder(tf.keras.Model):

	def __init__(self, rnn_cell,output_inner_fn=None, output_outer_fn=None):
		super(DynamicRNNEncoder, self).__init__(self)
		self.cell = rnn_cell
		self.output_inner_fn = output_inner_fn
		self.output_outer_fn = output_outer_fn
  
	@tf.function
	def call(self, input_data, conditions):
		# [batch, time, features] -> [time, batch, features]
		input_data = tf.transpose(input_data, [1, 0, 2])
		#[batch]
		conditions = tf.transpose(conditions, [1, 2, 0])
		#outputs = tf.TensorArray(tf.float32, input_data.shape[0] + 1)
		next_state = self.cell.get_initial_state(input_data[0,:,:])
		outputs = tf.TensorArray(tf.float32, 0,dynamic_size=True, element_shape=next_state.shape)
		next_state = self.cell.get_initial_state(input_data[0,:,:])
		if input_data.shape[0] == None:
			l = 1
		else:
			l = input_data.shape[0]
		for i in tf.range(l):
			if i > 0:
				previous_states = outputs.identity()
				previous_states = previous_states.stack()

				mask = conditions[i]
				mask = tf.stack([mask for k in range(previous_states.shape[-1])],axis=2)
				masked = previous_states*mask
				aggregated = tf.math.reduce_sum(masked,axis=0)
				#prepare next 
				next_state = aggregated
			output, state = self.cell(input_data[i], next_state)
			#append to the TensorArray
			outputs = outputs.write(i, output)

		if self.output_inner_fn: #process the individual node states and aggregate the results 
			tmp = outputs.identity()
			out = tf.TensorArray(tf.float32, tmp.size())
			for i in range(tmp.size()):
				out = out.write(i,self.output_inner_fn(tmp.read(i)))
			aggregated = tf.math.reduce_sum(out.stack(), axis=0)
			if self.output_outer_fn:
				return self.output_outer_fn(aggregated)
			else:
				return aggregated
		else:
			return tf.transpose(outputs.stack(), [1, 0, 2]) #return the stacked array (2nd dim is not fixed!)
if __name__ == "__main__":
	X =  np.array([np.array([np.array([np.random.randint(0,10)]) for i in range(5)]) for k in range(1000)])
	A = np.array([np.tril(np.random.randint(0,2,[5,5]),k=-1) for i in range(1000)])
	y = np.array([np.random.randint(0,2) for i in range(1000)])

	#print(X)
	#print(y)

	inputs = tf.keras.Input(shape=(5,1))
	input_A = tf.keras.Input(shape=(5,5))
	rnn = DynamicRNNEncoder(recurrent_fixed.GRUCell(6),tf.keras.layers.Dense(50,"relu"))(inputs, input_A)
	dense = tf.keras.layers.Dense(100, activation="relu")(rnn)
	outputs = tf.keras.layers.Dense(1,activation=None)(dense)
	model = tf.keras.Model(inputs=[inputs, input_A], outputs=outputs, name='test_model')
	model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
	model.summary()
	model.fit([X,A], y, batch_size=2, epochs=10, validation_split=0.2)