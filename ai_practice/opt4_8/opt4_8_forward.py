#coding:utf-8
import tensorflow as tf

def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    tf.add_to_collection('losses',tf.contrib_layers.l2_regularizer(regularizer)(w))     #正则化需要w和一个正则化因子
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([2,11],regularizer)
    b1 = get_bias([11])

    w2 = get_weight([11,1],regularizer)
    b2 = get_bias([1])

    y1 = tf.nn.relu(tf.matmul(x,w1) + b1)   #除输出层外都要过激活函数
    y = tf.matmul(y1,w2) + b2
    return y

