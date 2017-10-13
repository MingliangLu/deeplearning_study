import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

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

#显示图片
def image_show(index,dataset):
    index = index
    if dataset == "train":
        plt.imshow(train_set_x_orig[index])
        print ("y = " + str(train_set_y[:, index]) + ", 它是一张" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' 图片。")
    elif dataset == "test":
        plt.imshow(test_set_x_orig[index])
        print ("y = " + str(test_set_y[:, index]) + ", 它是一张" + classes[np.squeeze(test_set_y[:, index])].decode("utf-8") +  "' 图片。")
    plt.show()

#sigmoid函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s

#初始化参数w,b
def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) #w为一个列向量 dim*1
    b = 0    
    return w, b

#Y_hat = WT*X + b,进行一次迭代
def propagate(w, b, X, Y):
    m = X.shape[1] #样本个数
    Y_hat = sigmoid(np.dot(w.T,X)+b)
    #成本函数 L(y^,y)=−(ylog y^+(1−y)log(1−y^))
    #损失函数 J(w,b)=1m∑i=1mL(y^(i),y(i))
    cost = -(np.sum(np.dot(Y,np.log(Y_hat).T)+np.dot((1-Y),np.log(1-Y_hat).T))) / m

    #dw = X * (Y^ - Y)
    dw = (np.dot(X,(Y_hat-Y).T)) / m
    db = (np.sum(Y_hat-Y)) / m
    
    cost = np.squeeze(cost)

#梯度
    grads = {"dw" : dw,
             "db" : db}

    return grads, cost             

#梯度下降最优解
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = [] #记录成本值

    for i in range(num_iterations):
        grad, cost = propagate(w, b, X, Y)
        dw = grad["dw"]
        db = grad["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print("迭代%i次过后的成本值:%f" %(i,cost))
        
    #最终参数
    params = {"w" : w,
                "b" : b}

    #最终梯度
    grads = {"dw" : dw,
            "db" : db}
            
    return params, grads, costs

#预测结果
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    Y_hat = sigmoid(np.dot(w.T,X)+b)

    for i in range(Y_hat.shape[1]):
        if(Y_hat[:,i]) > 0.5:
            Y_prediction[:,i] = 1
        else:
            Y_prediction[:,i] = 0
        
    return Y_prediction

#建立整个模型
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    #训练集准确度
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    #测试集准确度
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

    print("训练集识别准确度: {} %".format(train_accuracy))
    print("测试集识别准确度: {} %".format(test_accuracy))

    d = {"costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train" : Y_prediction_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations}
    return d



#初始化数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#image_show(2,"train")
m_train = train_set_x_orig.shape[0] #训练集样本个数m
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1] #图片像素大小

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #原始训练集的设为（12288*209）
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T #原始测试集设为（12288*50）

#使用库逻辑回归
print("使用库逻辑回归")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_set_x_std = sc.fit_transform(train_set_x_flatten.T)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0,random_state=0)
lr.fit(train_set_x_std, train_set_y.T) #根据给定的训练数据拟合模型
#预测测试集精准度
text_set_x_std = sc.fit_transform(test_set_x_flatten.T)
Y_prediction_test = lr.predict(text_set_x_std)
test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - test_set_y.reshape(50,))) * 100
print("精准度为: {} %".format(test_accuracy))

print("使用自制逻辑回归")
train_set_x = train_set_x_flatten/255. #将训练集矩阵标准化
test_set_x = test_set_x_flatten/255. #将测试集矩阵标准化
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.005, print_cost = False)
