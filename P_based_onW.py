import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import W_based_on_distance as W
from sklearn import metrics
from sklearn import preprocessing

def compute_P_with_S(S,cluster_num,learn_rate=0.5,mult_rate=0.9):
    [n,m]=np.shape(S)
    S2=(S+S.T)
    #P_ini=np.random.rand(cluster_num,n)
    list_loss = []
    P_ini = 0.001*(np.random.randint(0,1000,(cluster_num,n)))
    #P_ini = np.ones((cluster_num,n))/cluster_num
    for i in range(1,100):
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


if __name__ == '__main__':
    data=scio.loadmat("dataset/mnist_1000.mat")
    x=data['mnist_1000']
    y=data['label']
    x = preprocessing.normalize(x)
    W1=W.distance(x,x)
    K=W.neighboors(W1,10)
    P=compute_P_with_S(K,10)
    PI=(np.argmax(P,axis=0))
    print(PI)
    print(y[0])
    print (metrics.normalized_mutual_info_score(PI,y[0]))