"""
A simple feed-forward neural network, to demo
digit classification using MNIST dataset

Usage:
 - First download MNIST dataset from http://yann.lecun.com/exdb/mnist/
 - Copy extracted files to data folder in root
 - Run `mnist_data_loader.py`
 - Run this script now
"""

import numpy as np

from python_mnist_loader import MNIST


def sigmoid(x):
    """Sigmoid logistic function
    """
    return 1. / (1. + np.exp(-x))


def sigmoid_deriv(x):
    """Derivative of sigmoid function
    """
    return sigmoid(x) * (1. - sigmoid(x))

# load dataset
print('Loading Dataset')
img = np.loadtxt('data/img', delimiter=',', dtype=float)
img_t = np.loadtxt('data/img_t', delimiter=',', dtype=float)
lbl = np.loadtxt('data/lbl', delimiter=',', dtype=int)
lbl_t = np.loadtxt('data/lbl_t', delimiter=',', dtype=int)

X = img
X_test = img_t

# one-hot-encode y values
y = np.zeros((lbl.shape[0], 10))
for i in range(y.shape[0]):
    y[lbl[i]] = 1

# one-hot-encode y_test values
y_test = np.zeros((lbl_t.shape[0], 10))
for i in range(y_test.shape[0]):
    y_test[lbl_t[i]] = 1
y_test = y_test[:, np.newaxis]

# make data between 0 to 1
X = X / 255

# start training network
epochs = 50000

# Layers => [ 784, 900, 10 ]

# Init weight
print('Initialising weights')
w1 = 2 * np.random.random((900, 785)) - 1
w2 = 2 * np.random.random((10, 901)) - 1

T = 2
learning_rate = 0.2
lmda = 1e-5

print('training...')

# add bias
z1 = np.hstack((np.ones((X.shape[0], 1)), X))

for k in range(epochs):
    # feed-forward

    a2 = z1.dot(w1.T)
    z2 = sigmoid(a2)

    z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
    a3 = z2.dot(w2.T)
    z3 = sigmoid(a3)

    del_3 = z3 - y
    del_2 = np.dot(del_3, w2[:, 1:]) * sigmoid_deriv(a2)

    delta_2 = del_3.T.dot(z2)
    delta_1 = del_2.T.dot(z1)

    # regularisation and weight update
    w2 = w2 - learning_rate * lmda * delta_2
    w1 = w1 - learning_rate * lmda * delta_1

    if k % 500 == 0:
        print('epoch: %d' % k)

print()
print('Predicting...')

z1 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

a2 = z1.dot(w1.T)
z2 = sigmoid(a2)

z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
a3 = z2.dot(w2.T)
z3 = sigmoid(a3)

# calculate accuracy
H = np.zeros((z3.shape[0], 1))
for i in range(z3.shape[0]):
    H[i] = z3[i].argmax()

true_vals = 0
total = z3.shape[0]

for i in range(total):
    if H[i].astype(int) == y[i][0].astype(int):
        true_vals += 1

accuracy = (true_vals / total) * 100
print('Accuracy is: {0}'.format(accuracy))
