import numpy as np 
import pickle
import tensorflow as tf 

cifarDir = 'cifar-10-batches-py/'
cifar100Dir = 'cifar-100-python/'

def unpickle(file):
	with open(file, 'rb') as fo:
		cifarDict = pickle.load(fo, encoding='bytes')
	return cifarDict

dirs = ['data_batch_1','data_batch_2','data_batch_3',
	'data_batch_4','data_batch_5','test_batch']


def readCifar(train=True):
	batches = []
	labels = []
	if train:
		for i in range(5):
			dataDict = unpickle(cifarDir+dirs[i])
			dataD = dataDict[b"data"]
			dataL = dataDict[b"labels"]
			batches.append(dataD)
			labels.append(dataL)


	else:
		dataDict = unpickle(cifarDir+dirs[-1])
		dataD = dataDict[b"data"]
		dataL = dataDict[b"labels"]
		batches.append(dataD)
		labels.append(dataL)

	return batches, labels

def readCifar100(train=True):
	if train:
		data = unpickle(cifar100Dir+"train")
		dataD = data[b"data"]
		dataL = data[b"fine_labels"]
	else:
		data = unpickle(cifar100Dir+"test")
		dataD = data[b"data"]
		dataL = data[b"fine_labels"]
	return dataD, dataL


def Cifar10(train = True ):
	x, y = readCifar(train)

	X = np.asarray(x)
	b,m,n = X.shape
	X = X.reshape(b*m,3,32,32).transpose(0,2,3,1)

	y = np.asarray(y)
	b,l = y.shape
	y = y.reshape(b*l)
	Y = np.zeros((y.size, y.max()+1))
	Y[np.arange(y.size),y] = 1
	return X,Y

def Cifar100(train=True):
	x,y = readCifar100(train)

	X = np.asarray(x)
	m,n = X.shape
	X = X.reshape(m,3,32,32).transpose(0,2,3,1)

	y = np.asarray(y)
	Y = np.zeros((y.size, y.max()+1))
	Y[np.arange(y.size),y] = 1
	return X,Y


def mnist(train=True):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	if train:
		s,m,n=x_train.shape
		y = np.asarray(y_train)
		Y = np.zeros((y.size, y.max()+1))
		Y[np.arange(y.size),y] = 1
		return x_train.reshape(s,m,n,1), Y
	else:
		s,m,n=x_test.shape
		y = np.asarray(y_test)
		Y = np.zeros((y.size, y.max()+1))
		Y[np.arange(y.size),y] = 1
		return x_test.reshape(s,m,n,1), Y

