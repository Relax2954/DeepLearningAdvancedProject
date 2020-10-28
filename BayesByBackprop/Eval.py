from mxnet import nd
import mxnet as mx
ctx = mx.cpu()

##EVALUATION
def eval_accuracy(data_iter, net, layer_params, numerator = 0., denominator = 0.):
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        out = net(data, layer_params)
        predictions = nd.argmax(out, axis=1)
        valid_pred = nd.sum(predictions == label)
        numerator = numerator+ valid_pred
        denominator = denominator + data.shape[0]
    return (numerator / denominator).asscalar()