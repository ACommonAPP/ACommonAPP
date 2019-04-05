#coding utf-8

#定义常量
seed = 23455
BATCH_SIZE = 8
STEPS = 3000

import tensorflow as tf
import numpy as np

#随机初始化生成正态分布的权值
w1 = tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))

#定义数据和对应的标签
rng = np.random.RandomState(seed)
X = rng.rand(32,2)
Y = [[int(x0 + x1 < 1)] for (x0,x1) in X]
#输出原始数据和标签
print('X:\n',X)
print('Y:\n',Y)

#定义神经网络的输入
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

#处理输入的值和权值
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#损失函数和优化器
loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i%500 == 0:
            print('w1:\n',sess.run(w1))
            print('w2:\n',sess.run(w2))
            print('第',int(i/500),'轮的损失函数值为:',sess.run(loss,feed_dict = {x:X,y_:Y}))
            print('\n')

    print("训练后的w1:",sess.run(w1))



