# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.io as scio
from sklearn import metrics

def kmeans_clu(data, clunum):
    data = preprocessing.normalize(data)
    #调用kmeans类
    clf = KMeans(n_clusters=clunum)
    s = clf.fit(data)

    la= clf.labels_
    return la
def get_clunum(y):
    set_la = set()
    for uni in y[0]:
        set_la.add(uni)
    clunum=len(set_la)
    print("cluster number is " +str(clunum))
    return clunum

data=scio.loadmat("dataset/mnist_1000.mat")
x=data['mnist_1000']
y=data['label']
cluster_num=get_clunum(y)
print(x.shape)
print(y.shape)
#W=distance(x,x)
#print(W.shape)
y1=kmeans_clu(x,cluster_num)
print(y1)
print (metrics.normalized_mutual_info_score(y1,y[0]))