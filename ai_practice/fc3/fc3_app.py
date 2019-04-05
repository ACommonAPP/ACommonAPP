# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import fc3_backward
import fc3_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, fc3_forward.INPUT_NODE])
        y = fc3_forward.forward(x, None)    #None表示无正则化
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(fc3_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(fc3_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1

#对图像进行预处理，使它满足神经网络的输入要求
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))    #为使它符合实际要求，先转化为灰度图，然后再转化为矩阵的形式
    threshold = 50  #消除噪声的门槛值，若小于50则为0，大于50则为255
    #输入的图片是白底黑字，要转化为黑底白字
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application():
    testNum = input("input the number of test pictures:")       #输入要识别几张图片
    for i in range(testNum):
        testPic = raw_input("the path of test picture:")    #输入要识别的图片的地址
        testPicArr = pre_pic(testPic)   #预处理图片
        preValue = restore_model(testPicArr)    #用已经训练好的模型预测图片
        print("The prediction number is:", preValue)


def main():
    application()


if __name__ == '__main__':
    main()
