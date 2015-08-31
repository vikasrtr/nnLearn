"""
Convert MNIST data files to numpy files

Usage:
First download MNIST dataset from http://yann.lecun.com/exdb/mnist/
Copy extracted files to data folder in root

Run this file to generate dataset
"""

import numpy as np
from python_mnist_loader import MNIST

# load dataset
loader = MNIST('data/')

# convert to numpy arrays
imgs, lbls = loader.load_training()
imgs_test, lbls_test = loader.load_testing()

print('Loading data...')
# save only 5000 training sets and 500 test sets

img = np.zeros((5000, 28 * 28), dtype=int)
for i in range(5000):
    img[i] = np.array(imgs[i])

lbl = np.zeros((5000, 1), dtype=int)
for i in range(5000):
    lbl[i] = np.array(lbls[i])

img_t = np.zeros((500, 28 * 28), dtype=int)
for i in range(500):
    img_t[i] = np.array(imgs_test[i])

lbl_t = np.zeros((500, 1), dtype=int)
for i in range(500):
    lbl_t[i] = np.array(lbls_test[i])

# save arrays
print('Saving data...')
np.savetxt('data/img', img, delimiter=',', fmt='%d')
np.savetxt('data/lbl', lbl, delimiter=',', fmt='%d')
np.savetxt('data/img_t', img_t, delimiter=',', fmt='%d')
np.savetxt('data/lbl_t', lbl_t, delimiter=',', fmt='%d')
