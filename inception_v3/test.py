import tensorflow as tf
import forward
import backward
import generateds

TEST_NUM = 100
TES_INTERVAL_SECS = 5

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None,
                                        backward.IMAGE_SIZE,
                                        backward.IMAGE_SIZE,
                                        backward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.inception_v3(x,is_training=False)

        saver = tf.train.Saver()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img,label = generateds.read_tfRecord(generateds.tfRecord_test)
        img_batch, label_batch = generateds.get_tfrecord(TEST_NUM, img, label)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()  # 3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4

                    xs, ys = sess.run([img_batch, label_batch])  # 5

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})

                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                    coord.request_stop()  # 6
                    coord.join(threads)  # 7

                else:
                    print('No checkpoint file found')
                    return


def main():
    test()  # 8


if __name__ == '__main__':
    main()
