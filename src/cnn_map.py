"""
CNN built from scratch
(c) Joris Teunisse 2019
"""

import tensorflow.keras.datasets.mnist as mnist
import numpy as np
import multiprocessing as mp

class CNN():
	def __init__(self, _data, _labels):
		# Initialize constants
		self.n_filters = 4
		self.filter_len = 5
		self.learning_rate = 0.01
		self.epochs = 25
		# Calculate feature map shapes
		self.fmap_len = (_data.shape[1] - (self.filter_len - 1))
		self.fmap_size = np.square(self.fmap_len)
		# Initialize lengths and weights of the fully connected layer
		self.fc_input_len = self.n_filters * self.fmap_size
		self.fc_output_len = np.unique(_labels).size
		self.weight_norm = 2 / (self.fc_input_len + self.fc_output_len)
		self.weights = np.random.randn(self.fc_input_len, self.fc_output_len) * self.weight_norm
		# Initialize filters
		self.filter_norm = 2 / (_data.shape[1] + self.fmap_size)
		self.filters = [np.random.randn(self.filter_len, self.filter_len) * self.filter_norm for _ in range(self.n_filters)]

	# Convolve a square filter around an input and return the resulting matrix
	def convolve2d(self, _if):
		# Calculate output matrix length
		out_len = _if[0].shape[0] - _if[1].shape[0] + 1
		# Initialize output matrix
		out = np.zeros((out_len, out_len))
		for i in range(out_len):
			for j in range(out_len):
				# Every output cell is the sum of an input subset multiplied by the filter
				out[i][j] = np.sum(_if[0][i:i+_if[1].shape[0], j:j+_if[1].shape[0]] * _if[1])
		return out

	# Predict a label for every data object: if training, backpropagate the results
	def predict(self, _data, _labels, _train):
		# Initialize performance metrics
		loss = 0
		acc = 0
		pool = mp.Pool(processes=self.n_filters)
		for i in range(_data.shape[0]):
			# TODO
			feature_maps = pool.map(self.convolve2d, [(_data[i], self.filters[f]) for f in range(self.n_filters)])
			relu_maps = [(x >= 0) * x for x in feature_maps]
			# Merge all maps into a vector to use as input layer
			fc_input = np.array(relu_maps).flatten()
			# Calculate input of output layer
			fc_output_in = np.dot(fc_input, self.weights)
			# Calculate output layer output using TanH for nonlinearity
			fc_output_out = np.tanh(fc_output_in)
			# Create target vector
			target = np.zeros(self.fc_output_len)
			target[_labels[i]] = 1
			# Update performance metrics
			loss += float(np.sum(np.square(target - fc_output_out)))
			acc += (np.argmax(fc_output_out) == np.argmax(target))
			# Print stats periodically
			if i % 100 == 0:
				print(i, "{:.3f}".format(acc / (i + 1)), "{:.3f}".format(loss / (i + 1)), end='\r')
			# Backpropagate results if training
			if _train:
				# Calculate loss derivative w.r.t. output layer output
				d_output_out = 2 * (fc_output_out - target)
				# Calculate derivative of output layer output w.r.t. its input
				d_output_in = d_output_out * (1 - np.square(fc_output_out))
				# Calculate derivative of output layer input w.r.t the input layer
				d_input = np.dot(self.weights, d_output_in)
				# Update weights by calculating the derivative of the output layer input w.r.t. them
				self.weights -= self.learning_rate * np.outer(fc_input, d_output_in)
				# Reshape deltas to match feature maps
				d_input_2d = d_input.reshape(self.n_filters, self.fmap_len, self.fmap_len)
				# TODO
				d_maps = [(feature_maps[f] >= 0) * d_input_2d[f] for f in range(self.n_filters)]
				filter_updates = pool.map(self.convolve2d, [(_data[i], d_maps[f]) for f in range(self.n_filters)])
				self.filters = [self.filters[f] - (self.learning_rate * filter_updates[f]) for f in range(self.n_filters)]

if __name__ == '__main__':
	# Load MINST data
	(train_d, train_l), (test_d, test_l) = mnist.load_data()
	# Normalize data
	train_d = train_d.astype('float32') / 255.0
	test_d = test_d.astype('float32') / 255.0
	# Initialize CNN
	cnn = CNN(train_d, train_l)
	for ep in range(cnn.epochs):
		print("Epoch", ep)
		# Train the network
		cnn.predict(train_d, train_l, True)
		print()
		# Test the network
		cnn.predict(test_d, test_l, False)
		print()