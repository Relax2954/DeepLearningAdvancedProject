# Reproduction of "A Simple Baseline for Bayesian Uncertainty in Deep Learning"

["A Simple Baseline for Bayesian Uncertainty in Deep Learning"](https://arxiv.org/pdf/1902.02476.pdf) by W. Maddox et al. is reproduced using Tensorflow, the 
dependencies used are shown in requirements.txt.

CIFAR10 and CIFAR100 datasets were extracted from https://www.cs.toronto.edu/~kriz/cifar.html, the folder names of each is expected to be cifar-10-batches-py and
cifar-100-python respectively respectively and located inside the SWAG folder. For MNIST dataset, data is extracted from tensorflow.

To pre-train the network:
```
python train.py
```
Then, to run the SWAG algorithm:
```
python SWAG.py
```
Finally to test the SWAG algorithm by extracting 30 samples from the approximated distribution:
```
python SWAGtest.py
```
To test the traditional SGD solution:
```
python test.py
```
