# define function for evaluating MLP
from mxnet import nd
from ActivationFunction import *

def net(X, layer_params):
    for i in range(len(layer_params) // 2 - 2):
        h_linear = nd.dot(X, layer_params[2*i]) + layer_params[2*i + 1]
        X = relu(h_linear)
    output = nd.dot(X, layer_params[-2]) + layer_params[-1]
    return output


def networkWeightShapes(num_hidden, num_outputs, num_inputs, num_layers):
    layer_par_shapes = []
    for i in range(num_layers + 2):
        if i == 0:  # first l
            W_shape = (num_inputs, num_hidden)
            b_shape = (num_hidden,)
        elif i == num_layers:  # end l
            W_shape = (num_hidden, num_outputs)
            b_shape = (num_outputs,)
        elif i == num_layers+1:
            continue
        else:
            W_shape = (num_hidden, num_hidden)
            b_shape = (num_hidden,)
        layer_par_shapes.extend([W_shape, b_shape])
    return layer_par_shapes

