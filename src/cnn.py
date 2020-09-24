"""
From-scratch CNN
(c) Joris Teunisse 2019
"""

import tensorflow.keras.datasets.mnist as mnist
import numpy as np

class CNN():
	def __init__(self, _data, _labels):
		# Initialize constants
		self.n_filters = 2
		self.filter_len = 3
		self.learning_rate = 0.01
		self.epochs = 25
		# Calculate feature map length
		self.fmap_len = (_data.shape[1] - (self.filter_len - 1))
		# Calculate fully connected layer lengths
		self.fc_input_len = self.n_filters * np.square(self.fmap_len)
		self.fc_output_len = np.unique(_labels).size
		# Initialize weights
		self.weight_norm = 2 / (self.fc_input_len + self.fc_output_len)
		self.weights = np.random.randn(self.fc_input_len, self.fc_output_len) * self.weight_norm
		# Initialize filters
		self.filter_norm = 2 / (np.square(_data.shape[1]) + np.square(self.fmap_len))
		self.filters = [np.random.randn(self.filter_len, self.filter_len) * self.filter_norm for _ in range(self.n_filters)]

	# Convolve a square filter around an input and return the resulting matrix
	def convolve2d(self, _input, _filter):
		# Calculate output matrix length
		out_len = _input.shape[0] - _filter.shape[0] + 1
		# Initialize output matrix
		out = np.zeros((out_len, out_len))
		for i in range(out_len):
			for j in range(out_len):
				# Every output cell is the sum of an input subset multiplied by the filter
				out[i][j] = np.sum(_input[i:i+_filter.shape[0], j:j+_filter.shape[0]] * _filter)
		return out

	# Predict a label for every data object: if training, backpropagate the results
	def predict(self, _data, _labels, _train):
		# Initialize stats
		loss = 0
		acc = 0
		for i in range(_data.shape[0]):
			# Convolve filters around the datum to create feature maps
			feature_maps = [self.convolve2d(_data[i], self.filters[f]) for f in range(self.n_filters)]
			# Add nonlinearity by using ReLU on the filters
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
			# Update stats
			loss += float(np.sum(np.square(target - fc_output_out)))
			acc += (np.argmax(fc_output_out) == np.argmax(target))
			# Print stats periodically
			if i % 100 == 99:
				print(i + 1, "{:.3f}".format(acc / (i + 1)), "{:.3f}".format(loss / (i + 1)), end='\r')
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
				for f in range(self.n_filters):
					# Calculate derivative of the input layer w.r.t a single map
					d_map = (feature_maps[f] >= 0) * d_input_2d[f]
					# Update filter using the derivative of the map w.r.t it
					self.filters[f] -= self.learning_rate * self.convolve2d(_data[i], d_map)
		# Preserve final stats
		print()

if __name__ == '__main__':
	# Load MINST data
	(train_d, train_l), (test_d, test_l) = mnist.load_data()
	# Normalize data
	train_d = train_d / 255
	test_d = test_d / 255
	# Initialize CNN
	cnn = CNN(train_d, train_l)
	for ep in range(cnn.epochs):
		print("Epoch", ep)
		# Train the network
		cnn.predict(train_d, train_l, True)
		# Test the network
		cnn.predict(test_d, test_l, False)