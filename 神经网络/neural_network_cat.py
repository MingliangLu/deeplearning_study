#神经网络猫识别器，采用两层神经，神经元个数为4
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import h5py

#导入数据
#导入数据
def load_dataset():
    train_dataset = h5py.File(".\\logistic\\train_cat.h5","r") #读取训练数据，共209张图片
    test_dataset = h5py.File(".\\logistic\\test_cat.h5", "r") #读取测试数据，共50张图片
    
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #原始训练集（209*64*64*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #原始训练集的标签集（y=0非猫,y=1是猫）（209*1）
    
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #原始测试集（50*64*64*3
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #原始测试集的标签集（y=0非猫,y=1是猫）（50*1）
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0])) #原始训练集的标签集设为（1*209）
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0])) #原始测试集的标签集设为（1*50）
    
    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#sigmoid函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
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
    W2 = np.random.randn(1,n_h)*0.01 
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
    m = Y.shape[0]
    
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
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1,2)) #g'(x) = 1-(g(x))^2
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    
    grads = {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}
    
    return grads

#更新参数 
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def nn_cat_model(X, Y, n_h, num_iterations = 1000, print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #迭代计算最优参数
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters) 
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 0.0075)
        
        if print_cost and i % 100 == 0:
            print("循环%i次后的成本: %f" %(i, cost))

    return parameters

#预测结果
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.zeros((1,A2.shape[1]))
    for i in range(0, A2.shape[1]):
        if A2[0,i] > 0.5:
            predictions[0,i] = 1
        else:
            predictions[0,i] = 0
    
    return predictions

#初始化数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0] #训练集样本个数m
m_test = test_set_x_orig.shape[0]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #原始训练集的设为（12288*209）
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T #原始测试集设为（12288*50）
train_x = train_set_x_flatten/255.
test_x = test_set_x_flatten/255.
#将数据送入模型
parameters = nn_cat_model(train_x, train_set_y.T.squeeze(), 4, num_iterations = 2500, print_cost = True)
#预测
predictions = predict(parameters, test_x)
predictions_train = predict(parameters, train_x)
Y = test_set_y
print ('准确度: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%') #打印精确度
Y = train_set_y
print ('训练集准确度: %d' % float((np.dot(Y,predictions_train.T) + np.dot(1-Y,1-predictions_train.T))/float(Y.size)*100) + '%') #打印精确度