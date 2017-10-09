#logistic_regression.py

#导入用到的包
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#导入数据
def load_dataset():
    train_dataset = h5py.File("train_cat.h5","r") #读取训练数据，共209张图片
    test_dataset = h5py.File("test_cat.h5", "r") #读取测试数据，共50张图片
    
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

#sigmoid函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s

#初始化参数w,b
def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) #w为一个dim*1矩阵
    b = 0    
    return w, b

#初始化数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()