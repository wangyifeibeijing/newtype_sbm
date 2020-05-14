import numpy as np
import matplotlib.pyplot as plt

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
def compute_P_with_S(S,cluster_num,learn_rate=0.5,mult_rate=0.5):
    [n,m]=np.shape(S)
    S2=(S+S.T)
    #P_ini=np.random.rand(cluster_num,n)
    list_loss = []
    #P_ini = np.random.uniform(0, 2.0/cluster_num, size=(cluster_num,n))
    P_ini = np.ones((cluster_num,n))/cluster_num
    for i in range(1,200):
        PTP=P_ini.T.dot(P_ini)
        upper_part=P_ini.dot(S2)+2*P_ini
        lower_part=upper_part.dot(PTP)


        eps = np.finfo(lower_part.dtype).eps
        updata_indicator=(upper_part)/(lower_part+eps)
        P_ini=np.multiply(P_ini,np.power(((1-learn_rate)+learn_rate*updata_indicator),mult_rate))
        loss=np.trace(P_ini.dot(S).dot(P_ini.T))
        #loss = np.linalg.norm((S-(P_ini.T.dot(P_ini))))
        #loss = np.linalg.norm(P_ini.dot(P_ini.T))
        if(i>25):
            list_loss.append(loss)
        print(loss)
    plt.plot(list_loss)
    plt.show()
    return P_ini
S=np.random.uniform(0, 0.001, size=(100,100))
S=deal_S(S)
P=compute_P_with_S(S,4,0.5)
#print(P)