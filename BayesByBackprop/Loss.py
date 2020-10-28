from config import *
import mxnet as mx
from mxnet import nd
ctx = mx.cpu()
from GaussianPrior import *
from MixturePrior import *
batch_size = configs['batch_size']
num_train = configs['num_train']
num_batches = num_train / batch_size


def total_loss(output, params, mus, sigmas, label_one_hot, log_prior):
    log_likelihood_s = nd.sum(nd.nansum(label_one_hot * nd.log_softmax(output), axis=0, exclude=True))
    log_prior_pre_sum = []
    for param in params:
        log_prior_pre_sum.append(nd.sum(log_prior(param)))
    log_prior_sum = sum(log_prior_pre_sum)
    log_var_posterior_pre_sum = []
    for i in range(len(params)):
        log_var_posterior_pre_sum.append(nd.sum(log_gaussian(params[i], mus[i], sigmas[i])))
    log_var_posterior_sum = sum(log_var_posterior_pre_sum)
    total_loss = 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_s
    return total_loss