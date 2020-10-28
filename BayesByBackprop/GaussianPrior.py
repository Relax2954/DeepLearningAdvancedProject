from config import *
import mxnet as mx
from mxnet import nd
import numpy as np
ctx = mx.cpu()


def log_gaussian(x, mu, sigma):
    log_gaussian = -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
    return log_gaussian

def gaussian_prior(x):
    sum_of_gaussians = nd.sum(log_gaussian(x, 0., nd.array([configs['sigma_p']], ctx=ctx)))
    return sum_of_gaussians