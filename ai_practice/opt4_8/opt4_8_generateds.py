#coding:utf-8
import tensorflow as tf
import numpy as np

seed = 2
def generateds():
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300,2)
    Y_ = [(int(x0*x0 + x1*x1 < 2)) for (x0,x1) in X]
    Y_c = [['red' if y_ == 1 else 'blue'] for y_ in Y_]

    #将Y_reshape成n行1列的形式
    Y_ = np.vstack(Y_).reshape(-1,1)

    return X,Y_,Y_c

# print(X)
# print(Y_)
# print(Y_c)