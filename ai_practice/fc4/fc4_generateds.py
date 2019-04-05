import tensorflow as tf
import numpy as np
from PIL import Image
import os

#定义好输入图片和标签的路径
image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28

#生成一个名为tfRecordName的tfrecord文件，用writer进行操作
def write_tfRecord(tfRecordName, image_path, label_path):
    #当给定的数据图片是jpg图片时， 通过这种方法可以制作数据集
    writer = tf.python_io.TFRecordWriter(tfRecordName)  #tfRecordName表示要生成的tfrecord文件的名字，用这个名字定义一个writer
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()    #label_path中存的数据是（‘图片名称’：对应的类别下标），而且分多行存储
    f.close()
    for content in contents:
        value = content.split() #将图片名和对应的类别下标分开，形成列表
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1

        #将所有的图片和标签，用tf.train.Example中的features，封装到example中
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())   #将example序列化为字符串进行存储
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    #制作好的tfrecord数据集文件默认保存在data_path(./data/)中
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    #可以从几个tfrecord文件中读取tfrecord文件名到这个filename_queue中
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)  #tfRecord_path告知tfrecord文件放在哪里了
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) #读出的每一个样本保存在serialized_example中
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })   #标签和图片的键名应该和制作tfrecord时的键名相同
    img = tf.decode_raw(features['img_raw'], tf.uint8)  #从img-raw恢复图像矩阵到img中
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)  #整个过程使用了两个线程
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
