"""
Neural network to recognize handwritten digits
(c) Joris Teunisse 2019
"""

import tensorflow.keras.datasets.mnist as mnist
import numpy as np

# Sigmoid
def act_s(x, p):
	return (x * (1 - x)) if p else (1 / (1 + np.exp(-x)))

# TanH
def act_t(x, p):
	return (1 - np.square(x)) if p else (np.tanh(x))

# ReLU
def act_r(x, p):
	return np.sign(x) if p else [0 if y <= 0 else y for y in x]

# Leaky ReLU
def act_l(x, p):
	return [0.01 if y <= 0 else 1 for y in x] if p else [(0.01 * y) if y <= 0 else y for y in x]

# Convert labels to appropriate target vectors
def to_targets(labels, acts):
	target_dims = (labels.size, np.unique(labels).size)
	targets = -(np.ones(target_dims)) if (acts[-1] == 't') else np.zeros(target_dims)
	for i in range(targets.shape[0]):
		targets[i][labels[i]] = 1
	return targets

# Z-score normalization
def z_score(x):
	return (x - np.mean(x)) / np.std(x)

# Main class
class NeuralNetwork():
	def __init__(self, net, acts, alpha):
		# Initialize class variables
		self.acts = acts
		self.alpha = alpha
		self.final = len(net) - 1

		# Initialize weights and biases using normalized random weights
		norm = [2 / (net[i + 1] + net[i]) for i in range(self.final)]
		self.weights = [np.random.randn(net[i + 1], net[i]) * norm[i] for i in range(self.final)]
		self.biases = [np.random.randn(net[i + 1]) * norm[i] for i in range(self.final)]
	
	def backpropagate(self, target, layers):
		# Initialize delta of past iterations
		past_delta = np.array([])
		for i in range(self.final, 0, -1):
			# Apply derivative of this layer's activation function
			delta = eval("act_" + self.acts[i - 1] + "(layers[i], True)")
			if i == self.final:
				# Multiply with error derivative: learning rate is added here as well
				delta *= self.alpha * (2 * (layers[i] - target))
			else:
				# Multiply with derivatives of past iterations
				delta *= np.dot(past_delta, self.weights[i])

			# Update all relevant variables
			self.weights[i - 1] -= np.outer(delta, layers[i - 1])
			self.biases[i - 1] -= delta
			past_delta = delta

	def predict(self, data, targets, train, debug):
		# Initialize confusion matrix, loss and accuracy
		cm = np.zeros((targets.shape[1], targets.shape[1]))
		loss = 0
		acc = 0
		for i in range(data.shape[0]): 
			# Create input layer
			layers = [data[i]]
			for j in range(self.final):
				# Create additional layers by feeding forward
				x = np.dot(layers[j], self.weights[j].T) + self.biases[j]
				layers.append(eval("act_" + self.acts[j] + "(x, False)"))

			if train:
				# Backpropagate results
				self.backpropagate(targets[i], layers)
			else:
				# Update confusion matrix and accuracy
				cm[np.argmax(targets[i])][np.argmax(layers[self.final])] += 1
				if (np.argmax(targets[i]) == np.argmax(layers[self.final])):
					acc += 100 / data.shape[0]
			
			# Update MSE
			loss += np.sum(np.square(targets[i] - layers[self.final]))
			if debug and i % 1000 == 0:
				# Print weights and MSE
				np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})
				print("Train" if train else "Test", i, "- MSE", '{:.3f}'.format(loss / (i + 1)))
				print("Output", layers[self.final])
				print("Target", targets[i], "\n")

		if not train:
			# Print confusion matrix and accuracy
			np.set_printoptions(suppress=True)
			print(cm)
			print("Testing accuracy: {:.2f}%\n".format(acc))

if __name__ == '__main__':
	# Set constants
	hiddens = [128]
	acts = ['l', 't']
	debug = False
	xval = False
	epochs = 25
	folds = 5
	decay_rate = 0.98
	alpha = 0.0001

	# Load and preprocess data
	(train_d, train_l), (test_d, test_l) = mnist.load_data()
	train_d = z_score(train_d.reshape(train_d.shape[0], train_d.shape[1] * train_d.shape[2]))
	test_d = z_score(test_d.reshape(test_d.shape[0], test_d.shape[1] * test_d.shape[2]))
	train_t = to_targets(train_l, acts)
	test_t = to_targets(test_l, acts)

	# Determine network shape
	net = [train_d.shape[1]] + hiddens + [train_t.shape[1]]

	if xval:
		# K-fold cross-validation
		s_size = int(train_d.shape[0] / folds)
		for i in range(folds):
			print("Fold", i)
			s = np.s_[i * s_size : (i + 1) * s_size]
			nn = NeuralNetwork(net, acts, alpha)
			for j in range(epochs):
				print("Epoch", j)
				nn.predict(np.delete(train_d, s, 0), np.delete(train_t, s, 0), True, debug)
				nn.predict(test_d[s], test_t[s], False, debug)
				nn.alpha *= decay_rate
	else:
		# Measure test set improvement per training epoch
		nn = NeuralNetwork(net, acts, alpha)
		for i in range(epochs):
			print("Epoch", i)
			nn.predict(train_d, train_t, True, debug)
			nn.predict(test_d, test_t, False, debug)
			nn.alpha *= decay_rate