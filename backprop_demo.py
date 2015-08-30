"""
NN simple

"""
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1. - sigmoid(x))

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([1, 1, 0, 0]).reshape(4, 1)

epochs = 10000

w1 = 2 * np.random.random((2, 3)) - 1
w2 = 2 * np.random.random((1, 3)) - 1

T = 2

for k in range(epochs):
    # feed-forward
    z1 = np.hstack((np.ones((X.shape[0], 1)), X))

    a2 = z1.dot(w1.T)
    z2 = sigmoid(a2)

    z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
    a3 = z2.dot(w2.T)
    z3 = sigmoid(a3)

    del_3 = z3 - y
    del_2 = np.dot(del_3, w2[:, 1:]) * sigmoid_deriv(a2)

    delta_2 = del_3.T.dot(z2)
    delta_1 = del_2.T.dot(z1)

    w2 = w2 - 0.2 * delta_2
    w1 = w1 - 0.2 * delta_1

    if k % 1000 == 0:
        print('epoch: %d' % k)

print('Predicting')
X_test = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

z1 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

a2 = z1.dot(w1.T)
z2 = sigmoid(a2)

z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
a3 = z2.dot(w2.T)
z3 = sigmoid(a3)

print(X_test)
print(z3)
