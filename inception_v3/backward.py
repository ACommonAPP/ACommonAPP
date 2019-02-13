#coding:utf-8

import tensorflow as tf
import numpy as np
import forward
import generateds
import os
import time
slim = tf.contrib.slim

IMAGE_SIZE = 299
NUM_CHANNELS = 3
BATCH_SIZE = 32
STEPS = 2500
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "inception_v3_model"

def backward():
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
    y = forward.inception_v3(x,forward.OUTPUT_NODE) #is_traing默认为true
    global_step = tf.Variable(0,trainable=False)

    slim.losses.softmax_cross_entropy(y, y_)
    loss = slim.losses.get_total_loss(add_regularization_losses=True)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        int(STEPS/ BATCH_SIZE),
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    # 生成并读取tfrecord文件
    generateds.generate_tfRecord()
    img, label = generateds.read_tfRecord(generateds.tfRecord_train)
    x_batch, y_batch = generateds.get_tfrecord(BATCH_SIZE,img,label)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(STEPS):
            xs, ys = sess.run([x_batch,y_batch])
            # xs = np.ones([BATCH_SIZE, 299, 299, 3])
            # _ys = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * BATCH_SIZE
            # ys = np.array(_ys)
            start_time = time.time()
            _,loss_value,step = sess.run([train_step, loss, global_step], feed_dict={x:xs, y_:ys})
            end_time = time.time()
            print("After %d training step(s), loss on training batch is %g. cost time is %gmin" % (step, loss_value, (end_time-start_time)/60))
            if i == STEPS-1:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        coord.request_stop()
        coord.join(threads)
def main():
    backward()

if __name__ == '__main__':
    main()




