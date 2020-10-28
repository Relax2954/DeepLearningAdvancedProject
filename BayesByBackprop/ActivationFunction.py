from mxnet import nd

def relu(X):
    min = nd.zeros_like(X)
    return nd.maximum(X, min)