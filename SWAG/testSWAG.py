from data import Cifar10, Cifar100, mnist
from VGG import VGG16

import numpy as np
import tensorflow as tf 

from sklearn.metrics import accuracy_score, log_loss

def flatW(vgg, sess):
	w = []
	wDict = vgg.weights(sess)
	keys = sorted(wDict.keys())
	for k in keys:
		w.append(wDict[k].flatten())
	return np.concatenate(w)

def unflattenW(weights, dims):
	keys = sorted(dims.keys())
	w = {}
	idx = 0
	for k in keys:
		d = dims[k]
		size = np.prod(d)
		values = weights[idx:idx+size]

		idx += size
		w[k] = values.reshape(d)
	return w


xTest, yTest = Cifar100(train=False)
#xTest, yTest = Cifar10(train=False)
#xTest, yTest = mnist(train=False)
_,w,h,nChannels = xTest.shape
nSamples, nClasses = yTest.shape

sess = tf.Session()

xInput = tf.placeholder(tf.float32, [None, w, h, nChannels])
yInput = tf.placeholder(tf.float32, [None, nClasses])

vgg = VGG16(xInput,0,nClasses, train=False)
logits = vgg.fc8
dim = vgg.Dims

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	labels=yInput, logits=logits))
predictions = tf.nn.softmax(logits)
acc = tf.equal(tf.argmax(predictions,1), tf.argmax(yInput,1))
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))

y = np.zeros((nSamples, nClasses))

parameters = np.load("SWAG.npz")
print(list(parameters.keys()))
Samples = 30
batchSize = 128
ystat = np.zeros((Samples, nSamples, nClasses))
validationL, validationA = [], []

for z in range(Samples):
	print("Sample ", z, " of ", Samples)
	w = parameters["Theta"]
	K = parameters["K"]
	sigma = parameters["Sigma"]
	D = parameters["D"]

	print(w.shape,sigma.shape, K, D.shape)

	z1 = np.random.normal(np.zeros((w.shape[0], )), np.ones((w.shape[0], )))
	z2 = np.random.normal(np.zeros((K, )), np.ones((K, )))

	sigmaSwag = np.clip(sigma, a_min=1e-30, a_max=None)
	wSample = w + (1/np.sqrt(2) * np.multiply(np.sqrt(sigmaSwag), z1)) + (1/np.sqrt(2*(K-1)) * np.dot(D,z2))

	weights = unflattenW(wSample, dim)
	vgg.loadW(weights, sess)

	step = np.ceil(nSamples/batchSize)


	for s in range(int(step)):
		xBatch = xTest[s*batchSize: (s+1)*batchSize]
		yBatch = yTest[s*batchSize: (s+1)*batchSize]

		pred = sess.run(predictions, feed_dict={xInput: xBatch, yInput: yBatch})
		y[s*batchSize: (s+1)*batchSize, :] += pred
		ystat[z, s*batchSize: (s+1)*batchSize, :] += pred

	lossV, accV = sess.run([crossEntropy, accuracy], feed_dict={xInput: xTest, yInput: yTest})
	validationL.append(lossV)
	validationA.append(accV)
	yPred = (1/Samples) * y

print("Accuracy: ", np.mean(accV), " +- ", np.sqrt(accV))
print("Loss: ", np.mean(lossV), "+-", np.sqrt(lossV))
	

accTest = accuracy_score(y_true=np.argmax(yTest, axis=1), y_pred=np.argmax(yPred, axis=1))
print("Test Accuracy: ", accTest)

