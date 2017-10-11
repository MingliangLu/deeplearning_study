import numpy as np

x=np.arange(1,4)
y=np.arange(2,5)
xx, yy = np.meshgrid(x,y)
Z = np.c_[xx.ravel(),yy.ravel()]
print(Z)