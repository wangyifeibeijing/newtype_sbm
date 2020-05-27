import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from sklearn.cluster import KMeans
from multiprocessing.dummy import Pool as ThreadPool
def distance_based_on_norm(x,y):
    [n,m]=x.shape
    [k,l]=y.shape
    W=np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            W[i,j]=np.linalg.norm(x[i] - y[j])
    return W
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

def distanceZ(a,b):
    a=np.mat(a.tolist())
    b=np.mat(b.tolist())

    [n,m]=a.shape
    [l,k] = b.shape
    b=b.T
    asum=np.power(a,2)

    aline=(asum.sum(axis=1))

    atemp=np.tile(aline, (1,l))


    bsum = np.power(b,2)
    bline = (bsum.sum(axis=0))
    btemp = np.tile(bline, (n,1))

    abtemp=a.dot(b)
    distance=atemp+btemp-abtemp-abtemp
    print(np.linalg.norm(a[0]))
    print(atemp)
    print("=====================================================================")
    print(btemp)
    print("=====================================================================")
    print(2*abtemp)
    return distance

def neighboors(W,neighnum=5):
    #W.dtype = 'float'
    K=np.sort(W)
    [n, m] = W.shape

    Ktemp = ((np.tile(K[:,neighnum+1], (1,m))))
    print(Ktemp)
    #eps = 1e-20
    eps = np.finfo(K.dtype).eps
    W_n=-1.0*(W/(Ktemp+eps))
    S = np.exp(W_n)
    S[S<np.exp(-1)]=0
    return  S

def capture_anchor(X,anchornum=100):
    #data = preprocessing.normalize(X)
    # 调用kmeans类
    clf = KMeans(n_clusters=anchornum)
    s = clf.fit(X)
    anchors=clf.cluster_centers_
    return anchors

if __name__ == '__main__':

    data=scio.loadmat("dataset/mnist_1000.mat")
    x=data['mnist_1000']
    #y=data['label']
    anchors=capture_anchor(x,10)
    W=distanceZ(x,anchors)
    print(W.shape)
    print(W)
    


    print("--------------------------------------------")
    K=neighboors(W,2)
    print(K)
    print(K[0])
    
    '''

    X=np.mat([[1,2,3],[3,-2,5],[3,3,6],[7,5,9]])
    Y=np.mat([[1,2,3],[10,3,4],[3,-2,5]])
    print(X)
    print(Y)
    W = distanceZ(X, Y)
    print(W)
    '''



