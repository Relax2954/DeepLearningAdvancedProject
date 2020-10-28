from config import *
import mxnet as mx
from mxnet import nd
import numpy as np
ctx = mx.cpu()


def gaussian(x, mu, sigma):
    return (1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))) * nd.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

def scale_mixture_prior(x):
    sigma_p1 = nd.array([configs['sigma_p1']], ctx=ctx)
    sigma_p2 = nd.array([configs['sigma_p2']], ctx=ctx)

    gaussian_1 = configs['pi'] * gaussian(x, 0., sigma_p1)
    gaussian_2 = (1 - configs['pi']) * gaussian(x, 0., sigma_p2)

    return nd.log(gaussian_1 + gaussian_2)