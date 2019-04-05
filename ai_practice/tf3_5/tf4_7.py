import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
seed = 2
BATCH_SIZE = 30

#基于seed产生随机数
rdm = np.random.RandomState(seed)
#随机返回300行2列的矩阵，表示300个（x0,x1)的点
X = rdm.randn(300,2)
#产生标签，平方和小于1的，对应的y表示1
Y_ = [(int(x0*x0 + x1*x1 < 2)) for (x0,x1) in X]    #为什么Y_用列形式表示就不能画出正确的红蓝图像，只有红色的图像
#遍历Y中每一个元素，1赋值红，0赋值蓝
Y_c = [['red' if y else 'blue'] for y in Y_]

print(X)
print(Y_)
print(Y_c)

#将Y_整理成n行1列的形式
Y_ = np.vstack(Y_).reshape(-1,1)
plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
plt.show()

#定义神经网络的输入，输出和反向传播方法
def get_weight(shape,regularizer):      #regularizer是正则化权重
    w = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape = shape))
    return b
x = tf.placeholder(tf.float32,shape = (None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])     #b1是11列吗？
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2)+b2

#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#定义反向传播方法
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
#没有正则化的过程
with tf.Session() as sess:
    STEPS = 40000
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = i*BATCH_SIZE % 300
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse,feed_dict = {x:X,y_:Y_})
            print('After', i, 'steps,loss is', loss_mse_v)
            print('\n')
    xx,yy = np.mgrid[-3:3:.01,-3:3:.01]
    grid = np.c_[xx.ravel(),yy.ravel()]
    probs = sess.run(y,feed_dict = {x:grid})
    probs = probs.reshape(xx.shape)
    print('w1:\n',sess.run(w1))
    print('b1:\n',sess.run(b1))
    print('w2:\n',sess.run(w2))
    print('b2:\n',sess.run(b2))
    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels = [.5])
    plt.show()

    # 有正则化的过程
    #定义反向传播方法
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
    with tf.Session() as sess:
        STEPS = 40000
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = i * BATCH_SIZE % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_mse_total_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print('After' ,i, 'steps,loss is', loss_mse_total_v)
                print('\n')
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)
        print('w1:\n', sess.run(w1))
        print('b1:\n', sess.run(b1))
        print('w2:\n', sess.run(w2))
        print('b2:\n', sess.run(b2))
        plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
        plt.contour(xx, yy, probs, levels=[.5])
        plt.show()


