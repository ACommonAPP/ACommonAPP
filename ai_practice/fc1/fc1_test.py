#coding:utf-8

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import fc1_forward
import fc1_backward

TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape = (None, fc1_forward.INPUT-DATA))
        y_ = tf.placeholder(tf.float32, shape = (None, fc1_forward.OUTPUT_DATA))
        y = fc1_forward.forward(x, None)    #此处没有正则化

        #实例化带滑动平均的saver对象
        ema = tf.train.ExponentialMovingAverage(fc1_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(fc1_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
                    print('after',global_step,'steps, test accuracy is',accuracy_score)
                else:
                    print('no checkpoint file is found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot = True)
    test(mnist)

if __name__ == '__main__':
    main()