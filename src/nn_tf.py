import tensorflow.keras as K

if __name__ == '__main__':
	(train_d, train_l), (test_d, test_l) = K.datasets.mnist.load_data()
	train_d = train_d.reshape(train_d.shape[0], train_d.shape[1] * train_d.shape[2]) / 255
	test_d = test_d.reshape(test_d.shape[0], test_d.shape[1] * test_d.shape[2]) / 255
	train_t = K.utils.to_categorical(train_l, 10)
	test_t = K.utils.to_categorical(test_l, 10)

	model = K.models.Sequential([K.layers.Dense(128, activation='relu', input_shape=(784,)),
								 K.layers.Dense(10, activation='sigmoid')])

	model.compile(loss=K.losses.mean_squared_error,
				  optimizer=K.optimizers.SGD(lr=0.0001),
				  metrics=['accuracy'])

	model.fit(train_d, train_t,
			  batch_size=1,
			  epochs=25,
			  validation_data=(test_d, test_t))

	model.evaluate(test_d, test_t)