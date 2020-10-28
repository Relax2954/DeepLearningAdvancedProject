from data import Cifar10, Cifar100, mnist
from VGG import VGG16

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def shuffle(x,y):
	p = np.random.permutation(x.shape[0])
	x = x[p,:]
	y = y[p,:]
	return x,y

def plot(loss,lossval,acc,accVal):
	plt.figure()
	plt.plot(np.arange(len(loss)),loss,label='Loss training set')
	plt.plot(np.arange(len(lossval)),lossval, label = 'Loss validation set')
	plt.xlabel('Epochs')
	plt.title('Loss per epoch')
	plt.legend()
	plt.savefig('loss.png')

	plt.figure()
	plt.plot(np.arange(len(acc)),acc,label='Accuracy training set')
	plt.plot(np.arange(len(accVal)),accVal, label = 'Accuracy validation set')
	plt.xlabel('Epochs')
	plt.legend()
	plt.savefig('acc.png')
	plt.close('all') 

def schedule(epoch, lrInit, swagLr, swagStart):
	t = (epoch) / (swagStart)
	lrRatio = swagLr/lrInit

	if t<=0.5:
		factor = 1.0
	elif t<= 0.9:
		factor = 1.0 - (1.0 - lrRatio) * (t - 0.5) / 0.4
	else:
		factor = lrRatio
	return lrInit * factor

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

config = ConfigProto()
config.gpu_options.allow_growth = True
tf.debugging.set_log_device_placement(True)

#X, Y = Cifar10(train=True)
X,Y = Cifar100(train=True)
#X, Y = mnist(train=True)

xTrain = X
yTrain = Y

#X, Y = Cifar10(train=False)
X,Y = Cifar100(train=False)
#X, Y = mnist(train=False)

xVal = X
yVal = Y

_,w,h,nChannels = xTrain.shape
nSamples, nClasses = yTrain.shape
print('Number of training samples: ',nSamples, ' Number of classes: ', nClasses)
print('Number of validation samples: ',xVal.shape[0])

sess = tf.Session(config=config)

xInput = tf.placeholder(tf.float32, [None, w, h, nChannels])
yInput = tf.placeholder(tf.float32, [None, nClasses])
learningRate = tf.placeholder(tf.float32, shape=[])


vgg = VGG16(xInput,0.5,nClasses)
logits = vgg.fc8

#Metrics
predictions = tf.nn.softmax(logits)
acc = tf.equal(tf.argmax(predictions,1), tf.argmax(yInput,1))
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))
validationL, validationL2, validationA = [], [], []
trainL, trainL2, trainA = [], [], []


#Training parameters
epochs = 200
#lrInit = 0.005 (for Mnist)
lrInit = 0.05
batchSize = 128

#Loss and Optimizer
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	labels=yInput, logits=logits))
L2term = tf.losses.get_regularization_loss()
optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=0.9)
train = optimizer.minimize(crossEntropy+L2term )


parameters = np.load("pretrain.npz")
init = tf.global_variables_initializer()
sess.run(init)
vgg.loadW(parameters, sess)

#SWAG
firstM = flatW(vgg,sess)
secondM = firstM **2
D = np.zeros((firstM.shape[0], 0))
swagN = 0
swagStart = 107
#swagLr = 0.001 (for Mnist)
swagLr = 0.01
swagModels = 20

for e in range(epochs):
	print("Epoch: ", e)
	lr = schedule(e, lrInit, swagLr, swagStart)
	for i in range(nSamples//batchSize):
		batchI = i*batchSize
		batchE = (i+1)*batchSize

		xBatch = xTrain[batchI:batchE]
		yBatch = yTrain[batchI:batchE]


		sess.run(train, feed_dict={xInput:xBatch, yInput: yBatch, learningRate: lr})

		if i%50 == 0:
			l, l2, a = sess.run([crossEntropy, L2term, accuracy], feed_dict={xInput: xBatch, yInput: yBatch})
			print("Iteration: ", i, " Loss = ", l, " LossL2 = ", l2," Acc= ", a, " LR= ", lr)

	if e >= swagStart:
		updatedParameters = flatW(vgg,sess)

		firstM = (swagN * firstM + updatedParameters) / (swagN+1)
		secondM = (swagN * secondM + updatedParameters**2) / (swagN + 1)

		if D.shape[1] == swagModels:
			D = np.delete(D,0,1)

		Dcol = updatedParameters - firstM
		D = np.append(D, Dcol.reshape(Dcol.shape[0],1), axis=1)

		swagN += 1


	lossV, loss2V, accV = sess.run([crossEntropy, L2term, accuracy], feed_dict={xInput: xVal, yInput: yVal})
	print("Loss Validatoin: ", lossV, " LossL2 Validatoin: ", loss2V, " Acc Validatoin: ", accV)	
	lossT, loss2T, accT = sess.run([crossEntropy, L2term, accuracy], feed_dict={xInput: xBatch, yInput: yBatch})

	validationL.append(lossV)
	validationL2.append(loss2V)
	validationA.append(accV)
	trainL.append(lossT)
	trainL2.append(loss2T)
	trainA.append(accT)


	xTrain, yTrain = shuffle(xTrain, yTrain)
	xVal, yVal = shuffle(xVal, yVal)
	
plot(trainL,validationL,trainA,validationA)
SWAGparam = {}
SWAGparam["Theta"] = firstM
SWAGparam["Sigma"] = secondM - firstM**2
SWAGparam["D"] = D
SWAGparam["K"] = swagModels

vggP = vgg.weights(sess)

np.savez("SWAG.npz",**SWAGparam)
np.savez("SGD.npz",**vggP)

