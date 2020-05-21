import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import preprocessing
from sklearn import metrics

from scipy import linalg as LA

def data_similarity(dataset,kneareast=5):
    dataset=preprocessing.normalize(dataset)
    #A=distance_based_on_norm(dataset,dataset)
    A = distance(dataset, dataset)
    #A=np.linalg.norm(dataset - dataset)
    A=0.5*(A+A.T)
    #print(A)
    sig=A.max()
    S=np.exp((-1*A)/2*sig)
    return S
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
    asum=a*a
    aline=(asum.sum(axis=1))
    atemp=np.tile(aline, (n,1))
    #print(atemp.shape)
    bsum = b*b
    bline = (bsum.sum(axis=1))
    btemp = np.tile(bline, (n, 1))
    btemp=btemp.T
    # print(atemp.shape)

    abtemp=a.dot(b.T)
    distance=atemp+btemp-2*abtemp
    return distance

def deal_S(S):
    [n,m ] = np.shape(S)
    one_vec=np.ones((n,1))
    DS=S.dot(one_vec).T[0,:]
    D=np.diag(np.power(DS,-0.5))
    #eps = np.finfo(D.dtype).eps
    #D_U=np.power(D+eps, -0.5)
    D_U=D
    S_RESULT=D_U.dot(S.dot(D_U))
    return S_RESULT
def compute_P_with_S(S,cluster_num,learn_rate=0.90,mult_rate=0.90):
    [n,m]=np.shape(S)
    S2=(S+S.T)
    #P_ini=np.random.rand(cluster_num,n)
    list_loss = []
    P_ini = 0.08*(np.random.randint(0,10,(cluster_num,n)))
    #P_ini = np.ones((cluster_num,n))/cluster_num
    for i in range(1,5000):
        PTP=P_ini.T.dot(P_ini)
        upper_part=P_ini.dot(S2)+2*P_ini
        lower_part=upper_part.dot(PTP)


        eps = np.finfo(lower_part.dtype).eps
        updata_indicator=(upper_part)/(lower_part+eps)
        P_ini=np.multiply(P_ini,np.power(((1-learn_rate)+learn_rate*updata_indicator),mult_rate))

        #loss=np.trace(P_ini.dot(S).dot(P_ini.T))
        loss = np.linalg.norm((S-(P_ini.T.dot(P_ini))))
        #II=np.identity(4)
        #loss = np.linalg.norm(P_ini.dot(P_ini.T)-II)
        if(i>0):
            list_loss.append(loss)

        #print(loss)
    plt.plot(list_loss)
    plt.show()
    return P_ini
def get_clunum(y):
    set_la = set()
    for uni in y[0]:
        set_la.add(uni)
    clunum=len(set_la)
    print("cluster number is " +str(clunum))
    return clunum
'''
S=np.random.uniform(0, 0.001, size=(100,100))
S=deal_S(S)
P=compute_P_with_S(S,4,0.5)
PI=(np.argmin(P,axis=0))
print(PI)

data=scio.loadmat("dataset/mnist_test.mat")
x=data['mnist_test']
y=data['mnist_label']
q=distance(x,x)
print(q.shape)
#print(P)
'''

data=scio.loadmat("dataset/mnist_1000.mat")
x=data['mnist_1000']
y=data['label']
cluster_num=get_clunum(y)
print(x.shape)
print(y.shape)
#W=distance(x,x)
#print(W.shape)
S=data_similarity(x)
#print(S)
#eigvalues, eigvectors = LA.eig(S)
#indices = np.argsort(eigvalues)[0:cluster_num]
#PINIT=eigvectors[:, indices]
#P_start=PINIT.T
#P_start[P_start < 0] = 0

#print(P_start.shape)
P=compute_P_with_S(S,cluster_num)

scio.savemat("ptemp.mat",{'P': P})

PI=(np.argmax(P,axis=0))
print(PI)
print(y[0])
print (metrics.normalized_mutual_info_score(PI,y[0]))


