import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='black', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')



iris = datasets.load_iris()  #提取iris数据集，shape为(150,4) 150个样本，每个样本四种属性
X = iris.data[:,[2,3]] #花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性，取其中两个(花瓣长度，花瓣宽度) (150,2)
y = iris.target #结果集  (150,)
#将数据集分为训练集(70%)和测试集(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 0)
#X_train(105,2) X_test(45,2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) #计算X_train的平均值和标准差用于之后的计算
X_trans_std = sc.transform(X_train) #执行标准化 (105,2)
X_test_std = sc.transform(X_test)             #(45, 2)
#堆叠 X_trans_std 和 X_test_std，为了后面的plot_decision_regions函数
X_combined_std = np.vstack((X_trans_std,X_test_std))    #(150,2)
y_combined = np.hstack((y_train,y_test))                #(150,)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_trans_std, y_train) #根据给定的训练数据拟合模型

#预测test database
y_pre = lr.predict_proba(X_test_std[0,:].reshape(1,2)) #预测第一个数据属于各个种类的概率
print(y_pre*100)

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

print(1)