import tensorflow as tf
import numpy as np
import recurrent_fixed

class DynamicRNN(tf.keras.Model):

	def __init__(self, rnn_cell):
		super(DynamicRNN, self).__init__(self)
		self.cell = rnn_cell
  
	@tf.function
	def call(self, input_data, test):
		# [batch, time, features] -> [time, batch, features]
		input_data = tf.transpose(input_data, [1, 0, 2])
		outputs = tf.TensorArray(tf.float32, input_data.shape[0])
		next_state = self.cell.get_initial_state(input_data[0,:,:])
		for i in tf.range(input_data.shape[0]):
			tf.print(test)
			#get the cell outputs
			output, state = self.cell(input_data[i], next_state)
			#appedn to the TensorArray
			outputs = outputs.write(i, output)
			#make a copy to work with
			previous_states = outputs.identity()
			#convert to tensor [times, batch,cell_dim]
			previous_states = previous_states.stack()
			#get the aggregations for each sample in the batch
			aggregated = tf.math.reduce_sum(previous_states,axis=0)
			#prepare next 
			next_state = aggregated
		return tf.transpose(outputs.stack(), [1, 0, 2]), next_state


X =  np.array([np.array([np.array([np.random.randint(0,10)]) for i in range(5)]) for k in range(1000)])
y = np.array([np.random.randint(0,2) for i in range(1000)])

#print(X)
#print(y)

inputs = tf.keras.Input(shape=(5,1))
rnn = DynamicRNN(recurrent_fixed.GRUCell(64))(inputs, "BAZINGA")
dense = tf.keras.layers.Dense(100, activation="relu")(rnn[1])
outputs = tf.keras.layers.Dense(1,activation=None)(dense)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test_model')
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
model.summary()
model.fit(X, y, batch_size=3, epochs=10, validation_split=0.2)