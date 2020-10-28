from __future__ import print_function
import numpy as np
from mxnet import nd, autograd
from matplotlib import pyplot as plt
import collections
import mxnet as mx
from config import *
from ActivationFunction import *
from Network import *
from Eval import *
from Loss import *

ctx = mx.cpu()

def data_transform(data, label):
    return data.astype(np.float32)/126.0, label.astype(np.float32)

mnist = mx.test_utils.get_mnist()
num_inputs = configs['num_inputs']
num_outputs = configs['num_outputs']
batch_size = configs['batch_size']
epochs = configs['epochs']
learning_rate = configs['learning_rate']

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=data_transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=data_transform),
                                     batch_size, shuffle=False)

num_train = configs['num_train']
num_batches = num_train / batch_size
num_layers = configs['num_hidden_layers']
num_hidden = configs['num_hidden_units']
layer_par_shapes = networkWeightShapes(num_hidden, num_outputs, num_inputs, num_layers)


def log_softmax_likelihood(yhat_linear, y):
    return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


weight_scale = configs['weight_scale']
rho_offset = configs['rho_offset']
mu_s = []
rho_s = []

for shape in layer_par_shapes:
    mu_s.append(nd.random_normal(shape=shape, ctx=ctx, scale=weight_scale))
    rho_s.append(rho_offset + nd.zeros(shape=shape, ctx=ctx))

variat_params = mu_s + rho_s

for param in variat_params:
    param.attach_grad()


def sample_eps(param_shapes):
    return [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]

def gaussian_samples_transform(sigmas, epsilons, mu_s):
    samples = []
    for v in range(len(mu_s)):
        samples.append(mu_s[v] + sigmas[v] * epsilons[v])
    return samples


smoothing_constant = .01
train_accuracy_list = []
test_accuracy_list = []

for epoc in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)

        with autograd.record():
            epsilons = sample_eps(layer_par_shapes)
            sigmas = []
            for rho in rho_s:
                sigmas.append(nd.log(1. + nd.exp(rho)))
            layer_params = gaussian_samples_transform(sigmas, epsilons, mu_s)
            output = net(data, layer_params)
            loss = total_loss(output, layer_params, mu_s, sigmas, label_one_hot, gaussian_prior)

        loss.backward()

        for param in variat_params:
            param[:] = param - learning_rate * param.grad

        current_loss = nd.mean(loss).asscalar()

        if ((i == 0) and (epoc == 0)):
            moving_loss = current_loss
        else:
            moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * current_loss


    test_accuracy = eval_accuracy(test_data, net, mu_s)
    train_accuracy = eval_accuracy(train_data, net, mu_s)
    train_accuracy_list.append(np.asscalar(train_accuracy))
    test_accuracy_list.append(np.asscalar(test_accuracy))
    print("Epoc %s. Loss: %s, Train_accuracy %s, Test_accuracy %s" %
          (epoc, moving_loss, train_accuracy, test_accuracy))