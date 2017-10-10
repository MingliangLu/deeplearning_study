import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

#生成样本数据
def load_planar_dataset():
    m = 400 #总样本数
    N = int(m/2) #每种样本数
    D = 2 #维数
    a = 4 #花瓣延伸的最大长度
    X = np.zeros((m,D)) #初始化样本矩阵
    Y = []
    
    #随机分配样本坐标，使样本组成一朵花形
    for j in range(2): 
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 #角度
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 #半径
        X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
        Y[:ix] = 'r' if j==0 else 'b'
    
    X = X.T
    Y = Y.T
    
    plt.scatter(X[0,:], X[1,:],c=0,s=40,cmap=plt.cm.Spectral)
    plt.show()
    return X,Y

load_planar_dataset()