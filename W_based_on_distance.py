import numpy as np
import scipy.io as scio
def distance(a,b):

    [n,m]=a.shape
    asum=a.dot(a.T)
    #asum=a*a
    #aline=(asum.sum(axis=1))
    aline=np.diag(asum)
    atemp=np.tile(aline, (n,1))

    bsum = b.dot(b.T)
    bline = np.diag(bsum)
    btemp = np.tile(bline, (n, 1))
    btemp=btemp.T

    abtemp=a.dot(b.T)

    distance=atemp+btemp-2*abtemp
    return distance
def neighboors(W,neighnum=5):
    #W.dtype = 'float'
    K=np.sort(W)
    [n, m] = W.shape

    Ktemp = ((np.tile(K[:,neighnum+1], (n, 1))).T)
    eps = 1e-20
    W_n=-1.0*(W/(Ktemp+eps))
    S = np.exp(W_n)
    S[S<np.exp(-1)]=0
    return  S
if __name__ == '__main__':
    data=scio.loadmat("dataset/mnist_1000.mat")
    x=data['mnist_1000']
    y=data['label']

    W=distance(x,x)
    K=neighboors(W)
    #print(W)
    print("--------------------------------------------")
    print(K)