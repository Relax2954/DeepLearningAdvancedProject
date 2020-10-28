from data import Cifar10, Cifar100,mnist
from VGG import VGG16

import numpy as np
import tensorflow as tf 

from sklearn.metrics import accuracy_score, log_loss

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

parameters = np.load("SGD.npz")
vgg.loadW(parameters, sess)

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	labels=yInput, logits=logits))
predictions = tf.nn.softmax(logits)
acc = tf.equal(tf.argmax(predictions,1), tf.argmax(yInput,1))
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))

y = np.zeros((nSamples, nClasses))
batchSize = 128

step = np.ceil(nSamples/batchSize)

for s in range(int(step)):
	xBatch = xTest[s*batchSize: (s+1)*batchSize]
	yBatch = yTest[s*batchSize: (s+1)*batchSize]

	pred = sess.run(predictions, feed_dict={xInput: xBatch, yInput: yBatch})
	y[s*batchSize: (s+1)*batchSize, :] += pred
	

lossTest = log_loss(y_true=yTest, y_pred=y)
accTest = accuracy_score(y_true=np.argmax(yTest, axis=1), y_pred=np.argmax(y, axis=1))
print("Test Loss: ", lossTest)
print("Test Accuracy: ", accTest)

