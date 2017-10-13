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

#sigmoid函数
def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

#获得每一层的节点数
def layer_sizes(X, Y):
    n_x = X.shape[0] #a[0]输入层
    n_h = 4          #a[1]中间层
    n_y = Y.shape[0] #a[2]输出层
    
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h,n_x)*0.01 #W1 shape (n_h,n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(1,n_h)*0.01 #为什么不是(n_h,1)
    b2 = np.zeros((1,1))

    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return parameters

#前向传播
def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}
    
    return A2, cache

#计算成本函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    cost = -np.sum(np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2)))/m
    cost = np.squeeze(cost)
    
    return cost

#反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 -Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1,2)) #g'(x) = 1-(g(x))^2
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    
    grads = {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}
    
    return grads
