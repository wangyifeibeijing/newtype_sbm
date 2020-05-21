import os
import struct
import numpy as np
import scipy.io as scio

def load_mnist(path, kind='t10k'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
def rand_sample(x,y,samnum=1000):
    [n,m]=x.shape
    aim=np.hstack((x, y))

    row_rand_array = np.arange(aim.shape[0])

    np.random.shuffle(row_rand_array)

    row_rand = aim[row_rand_array[0:samnum]]
    x_r=row_rand[:,0:m]
    y_r=row_rand[:,m]
    return [x_r,y_r]
'''
[d,l]=load_mnist("")
print(d.shape)
print(l.shape)
scio.savemat('mnist_test.mat', {'mnist_test': d,'mnist_label': l})
'''
data=scio.loadmat("mnist.mat")
x=data['mnist']
y=data['mnist_label']
[x1,y1]=rand_sample(x,y.T,samnum=1000)
print(y1)
scio.savemat('mnist_1000.mat', {'mnist_1000': x1,'label':y1})

'''

print(x.shape)
print(y.shape)
'''