import tensorflow as tf 
import numpy as np 

class VGG16(object):
	def __init__(self, x,prob,clases,train=True,sess=None):
		self.X = x
		self.nClasses = clases
		self.KeepProb = prob
		self.train = train
		self.params = {}

		self.Dims = {}

		self.create()

	def create(self):
		if self.train:
			xInput = tf.image.random_flip_left_right(self.X)
		else:
			xInput = self.X
		xInput = tf.image.resize(xInput, (32,32))
		if self.train:
			xInput = tf.image.resize_image_with_crop_or_pad(xInput, 32+4, 32+4)
			xInput = tf.random_crop(xInput, size=[tf.shape(xInput)[0],32,32,3])
			#xInput = tf.random_crop(xInput, size=[tf.shape(xInput)[0],32,32,1])
		xInput = xInput/255
		mean = tf.constant([0.485,0.456,0.406], dtype=tf.float32, shape=[1,1,1,3], name='mean')
		std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32, shape=[1,1,1,3], name='std')		mean = tf.constant([0.485,0.456,0.406], dtype=tf.float32, shape=[1,1,1,3], name='mean')
		xInput = (xInput-mean)/std

		conv1_1 = self.convLayer(xInput, 64, 'conv1_1')
		conv1_2 = self.convLayer(conv1_1, 64, 'conv1_2')
		pool1 = self.maxPool(conv1_2, 'pool1')

		conv2_1 = self.convLayer(pool1, 128, 'conv2_1')
		conv2_2 = self.convLayer(conv2_1, 128, 'conv2_2')
		pool2 = self.maxPool(conv2_2, 'pool2')

		conv3_1 = self.convLayer(pool2, 256, 'conv3_1')
		conv3_2 = self.convLayer(conv3_1, 256, 'conv3_2')
		conv3_3 = self.convLayer(conv3_2, 256, 'conv3_3')
		pool3 = self.maxPool(conv3_3, 'pool3')

		conv4_1 = self.convLayer(pool3, 512, 'conv4_1')
		conv4_2 = self.convLayer(conv4_1, 512, 'conv4_2')
		conv4_3 = self.convLayer(conv4_2, 512, 'conv4_3')
		pool4 = self.maxPool(conv4_3, 'pool4')

		conv5_1 = self.convLayer(pool4, 512, 'conv5_1')
		conv5_2 = self.convLayer(conv5_1, 512, 'conv5_2')
		conv5_3 = self.convLayer(conv5_2, 512, 'conv5_3')
		pool5 = self.maxPool(conv5_3, 'pool5')

		flat = tf.reshape(pool5, [-1, 1*1*512])
		drop6 = self.dropout(flat, self.KeepProb)
		fc6 = self.fcLayer(drop6, 1*1*512, 512, name='fc6')

		drop7 = self.dropout(fc6, self.KeepProb)
		fc7 = self.fcLayer(drop7, 512, 512, name='fc7')

		self.fc8 = self.fcLayer(fc7, 512, self.nClasses, relu=False, name='fc8')

		for k,v in self.params.items():
			self.Dims[k] = v.get_shape().as_list()


	def convLayer(self, x, numFilters, name, filterHeight=3, filterWidth=3, 
			stride=1, padding='SAME'):
		wd = 5e-4
		L2 = tf.contrib.layers.l2_regularizer(scale=wd)
		inputChannels = int(x.get_shape()[-1])
		with tf.variable_scope(name) as scope:
			W = tf.get_variable('weights', regularizer=L2, shape=[filterHeight, filterWidth, inputChannels, numFilters],
					initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_OUT'))
			b = tf.get_variable('biases',shape = [numFilters], initializer = tf.constant_initializer(0.0))
			print(x.get_shape(), W.get_shape())
			conv = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)

			z = tf.nn.bias_add(conv,b)

			a = tf.nn.relu(z)

			self.params[W.name] = W
			self.params[b.name] = b

			return a

	def maxPool(self, x, name, filterHeight=2, filterWidth=2, 
			stride=2, padding='VALID'):
		
		return tf.nn.max_pool(x, ksize=[1,filterHeight,filterWidth,1],
				strides=[1,stride,stride,1], padding=padding, name=name)

	def dropout(self, x, prob):
		return tf.nn.dropout(x, rate = prob)

	def fcLayer(self, x, inputSize, outputSize, name, relu = True):
		wd = 5e-4
		L2 = tf.contrib.layers.l2_regularizer(scale=wd)
		with tf.variable_scope(name) as scope:
			W = tf.get_variable('weights', regularizer=L2, shape=[inputSize, outputSize],
					initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_OUT'))#tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
			b = tf.get_variable('biases', shape=[outputSize], 
					initializer = tf.constant_initializer(0.0))

			z = tf.nn.bias_add(tf.matmul(x,W),b)
			
			self.params[W.name] = W
			self.params[b.name] = b

			if relu:
				a = tf.nn.relu(z)
				return a
			else:
				return z

	def weights(self, sess):
		return sess.run(self.params)

	def loadW(self, weights, sess):
		for k,v in weights.items():
			sess.run(self.params[k].assign(weights[k]))


