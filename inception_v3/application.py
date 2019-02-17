#coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import backward
import generateds
from labels import label


# def application(image_path):
# #     with tf.Graph().as_default() as g:
# #         x = tf.placeholder(tf.float32, [1,
# #                                         backward.IMAGE_SIZE,
# #                                         backward.IMAGE_SIZE,
# #                                         backward.NUM_CHANNELS])
# #         y = forward.inception_v3(x,is_training=False)
# #         prob = tf.nn.softmax(logits=y)
# #
# #         saver = tf.train.Saver()
# #
# #         img = generateds.pre_image(image_path)
# #
# #         with tf.Session() as sess:
# #             ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
# #             if ckpt and ckpt.model_checkpoint_path:
# #                 saver.restore(sess, ckpt.model_checkpoint_path)
# #
# #                 img_1 = sess.run(img)
# #                 probability = sess.run(prob,feed_dict={x:img_1})
# #
# #                 prob_sort = np.argsort(probability[0])[-1:-6:-1]
# #                 values = []
# #                 bar_label = []
# #                 for n,i in enumerate(prob_sort):
# #                     print(label[i],':',probability[0][i])
# #                     values.append(probability[0][i])
# #                     bar_label.append(label[i])
# #
# #                 fig = plt.figure(u"Top-3 预测结果")
# #                 ax = fig.add_subplot(111)
# #                 ax.bar(range(len(values)),values,tick_label=bar_label,width=0.5,fc='b')
# #                 ax.set_ylabel(u'probability')
# #                 ax.set_title(u'Top-5')
# #                 for a, b in zip(range(len(values)), values):
# #                     ax.text(a, b + 0.0005, b, ha='center', va='bottom', fontsize=7)
# #                 plt.show()
# #
# #             else:
# #                 print('No checkpoint file found')
# #                 return
# #
# # def main():
# #     img_path = input("please input the path of image:")
# #     application(img_path)  # 8
# #
# #
# # if __name__ == '__main__':
# #     main()




def application(image_path):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1,
                                        backward.IMAGE_SIZE,
                                        backward.IMAGE_SIZE,
                                        backward.NUM_CHANNELS])
        y = forward.inception_v3(x,is_training=False)
        prob = tf.nn.softmax(logits=y)

        saver = tf.train.Saver()

        img = generateds.pre_image(image_path)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                img_1 = sess.run(img)
                probability = sess.run(prob,feed_dict={x:img_1})

                prob_sort = np.argsort(probability[0])[-1:-6:-1]

        return prob_sort[0],probability[0][prob_sort[0]]
