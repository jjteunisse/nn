"""
From-scratch CNN for the MNIST database
(c) Joris Teunisse 2019
"""

import tensorflow.keras.datasets.mnist as mnist
import numpy as np

class CNN():
	def __init__(self, _data, _labels):
		# Initialize constants
		self.n_filters = [2, 3]
		self.filter_lens = [5, 3]
		self.learning_rate = 0.01
		self.epochs = 25
		# Calculate feature map lengths
		self.fmap_lens = [_data.shape[2] - self.filter_lens[0] + 1]
		self.fmap_lens.append(self.fmap_lens[0] - self.filter_lens[1] + 1)
		# Calculate fully connected layer lengths
		self.fc_input_len = self.n_filters[1] * np.square(self.fmap_lens[1])
		self.fc_output_len = np.unique(_labels).size
		# Initialize weights
		self.weight_norm = 2 / (self.fc_input_len + self.fc_output_len)
		self.weights = np.random.randn(self.fc_input_len, self.fc_output_len) * self.weight_norm
		# Initialize filters (TODO: does depth influence normalization? Assumption: no)
		self.filter_norms = [2 / (np.square(_data.shape[2]) + np.square(self.fmap_lens[0]))]
		self.filter_norms.append(2 / (np.square(self.fmap_lens[0]) + np.square(self.fmap_lens[1])))
		self.filters_0 = np.array([np.random.randn(1, self.filter_lens[0], self.filter_lens[0]) * self.filter_norms[0] for _ in range(self.n_filters[0])])
		self.filters_1 = np.array([np.random.randn(self.n_filters[0], self.filter_lens[1], self.filter_lens[1]) * self.filter_norms[1] for _ in range(self.n_filters[1])])
		# print(self.filters_0.shape, '\n', self.filters_0)
		# print(self.filters_1.shape, '\n', self.filters_1)

	# Convolve a (3d) filter around a (3d) input and return the resulting (2d) matrix
	def convolve3d(self, _input, _filter):
		# Calculate output matrix length
		i_len = _input.shape[1]
		f_len = _filter.shape[1]
		o_len = i_len - f_len + 1
		# Initialize output matrix
		out = np.zeros((o_len, o_len))
		for i in range(o_len):
			for j in range(o_len):
				# Every output cell is the sum of an input subset multiplied by the filter
				out[i][j] = np.sum(_input[:, i:i+f_len, j:j+f_len] * _filter)
		return out

	# Fully convolve a (3d) filter around a (2d) input and return the resulting (3d) tensor
	def full_convolve3d(self, _input, _filter):
		# Calculate output tensor length
		i_len = _input.shape[0]
		f_len = _filter.shape[1]
		o_len = i_len + f_len - 1
		f_depth = _filter.shape[0]
		# Initialize output tensor
		out = np.zeros((f_depth, o_len, o_len))
		# Create padded input
		padded_i_len = i_len + 2 * (f_len - 1)
		padded_input = np.zeros((padded_i_len, padded_i_len))
		padded_input[(f_len - 1):o_len, (f_len - 1):o_len] = _input.copy()
		for z in range(f_depth):
			for i in range(o_len):
				for j in range(o_len):
					out[z][i][j] += np.sum(_filter[z] * padded_input[i:i+f_len, j:j+f_len])
		return out

	# Predict a label for every data object: if training, backpropagate the results
	def predict(self, _data, _labels, _train):
		# Initialize stats
		loss = 0
		acc = 0
		for i in range(_data.shape[0]):
			# Convolve filters around the datum to create feature maps
			feature_maps_0 = [self.convolve3d(_data[i], self.filters_0[f]) for f in range(self.n_filters[0])]
			# Add nonlinearity by using ReLU on the filters
			relu_maps_0 = np.array([(x >= 0) * x for x in feature_maps_0])
			# Convolve filters around the relu maps to create the next feature maps
			feature_maps_1 = [self.convolve3d(relu_maps_0, self.filters_1[f]) for f in range(self.n_filters[1])]
			# Add nonlinearity by using ReLU on the filters
			relu_maps_1 = np.array([(x >= 0) * x for x in feature_maps_1])
			# Merge all maps into a vector to use as input layer
			fc_input = np.array(relu_maps_1).flatten()
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
				d_input_2d = d_input.reshape(self.n_filters[1], self.fmap_lens[1], self.fmap_lens[1])
				d_relu_1 = np.zeros_like(feature_maps_0)
				for f in range(self.n_filters[1]):
					# Calculate derivative of the input layer w.r.t a single map
					d_feature_1 = (feature_maps_1[f] >= 0) * d_input_2d[f]
					# TODO: backpropagate values
					d_relu_1 += self.full_convolve3d(d_input_2d[f], np.rot90(np.rot90(self.filters_1[f])))
					# Update filter using the derivative of the map w.r.t it
					self.filters_1[f] -= self.learning_rate * self.convolve3d(relu_maps_0, d_feature_1)
				for f in range(self.n_filters[0]):
					# Calculate derivative of the input layer w.r.t a single map
					d_feature_0 = (feature_maps_0[f] >= 0) * d_relu_1[f]
					# Update filter using the derivative of the map w.r.t it
					self.filters_0[f] -= self.learning_rate * self.convolve3d(_data[i], d_feature_0)
		# Preserve final stats
		print()

if __name__ == '__main__':
	# Load MINST data
	(train_d, train_l), (test_d, test_l) = mnist.load_data()
	# Normalize and 3d-ify data
	train_d = np.array([[x / 255] for x in train_d])
	test_d = np.array([[x / 255] for x in test_d])
	# Initialize CNN
	cnn = CNN(train_d, train_l)
	for ep in range(cnn.epochs):
		print("Epoch", ep)
		# Train the network
		cnn.predict(train_d, train_l, True)
		# Test the network
		cnn.predict(test_d, test_l, False)