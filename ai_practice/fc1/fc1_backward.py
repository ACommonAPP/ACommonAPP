#coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import fc1_forward

REGULARIZER = 0.0001
STEPS = 50000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model_1/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, shape = (None, fc1_forward.INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape = (None, fc1_forward.OUTPUT_NODE))
    y = fc1_forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable = False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        50000/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')

    #实例化saver对象
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # #实现断点续训
        # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        # if ckpt and ckpt.mpdel_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs, y_:ys})
            if i%1000 == 0:
                print('after',step,'steps, loss on training step is',loss_value)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot = True)
    backward(mnist)

if __name__ == '__main__':
    main()


