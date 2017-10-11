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
    Y = np.zeros((m,1), dtype='uint8') #初始化标签矩阵，0为红色，1为蓝色
    
    #随机分配样本坐标，使样本组成一朵花形
    for j in range(2): 
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 #角度
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 #半径
        X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
        Y[ix] = j
    
    X = X.T
    Y = Y.T
    Y = np.squeeze(Y)
    
    plt.scatter(X[0,:], X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)
    return X,Y

#生成分类器的边界
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1

    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0,:], X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)

X,Y = load_planar_dataset()
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel()) #数据拟合

plot_decision_boundary(lambda x:clf.predict(x),X,Y)

plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
print('logistic回归的精确度：%d' % float((np.dot(Y,LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) + "%")
plt.show()
