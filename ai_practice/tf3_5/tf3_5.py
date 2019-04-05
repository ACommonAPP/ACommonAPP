#codeing utf-8
#两层简单神经网络（全连接）
import tensorflow as tf

#定义输入和参数
#用placeholder 定义输入
x = tf.placeholder(tf.float32,shape=(None,2))
w1 = tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#调用会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result of tf3_5.py is:\n",sess.run(y,feed_dict={x:[[0.1,0.2],
                                                                  [0.2,0.3],[0.3,0.4]]}))
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))

