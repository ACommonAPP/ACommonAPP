import tensorflow as tf
import numpy as np
from PIL import Image
import forward
import os

image_train_path = './data/train/images/'
label_train_path = './data/train/labels.txt'
tfRecord_train = './data/inception_v3_train.tfrecords'
image_test_path = './data/test/images/'
label_test_path = './data/test/labels.txt'
tfRecord_test = './data/inception_v3_test.tfrecords'
data_path = './data'

def write_tfRecord(tfRecordName, image_path, label_path):
    #当给定的数据图片是jpg图片时， 通过这种方法可以制作数据集
    writer = tf.python_io.TFRecordWriter(tfRecordName)  #tfRecordName表示要生成的tfrecord文件的名字，用这个名字定义一个writer
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()    #label_path中存的数据是（‘图片名称’：对应的类别下标），而且分多行存储
    f.close()
    for content in contents:
        value = content.split(sep=' ') #将图片名和对应的类别下标分开，形成列表
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_1 = img.resize((299, 299), Image.ANTIALIAS)
        if np.array(img_1).shape == (299,299,3):
            img_raw = img_1.tobytes()
            labels = [0] * forward.OUTPUT_NODE
            labels[int(value[1])] = 1

            # 将所有的图片和标签，用tf.train.Example中的features，封装到example中
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
            writer.write(example.SerializeToString())  # 将example序列化为字符串进行存储
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
    #write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    #write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    #可以从几个tfrecord文件中读取tfrecord文件名到这个filename_queue中
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)  #tfRecord_path告知tfrecord文件放在哪里了
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) #读出的每一个样本保存在serialized_example中
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([forward.OUTPUT_NODE], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })   #标签和图片的键名应该和制作tfrecord时的键名相同
    img = tf.decode_raw(features['img_raw'], tf.uint8)  #从img-raw恢复图像矩阵到img中
    image = tf.reshape(img, [299,299, 3])
    label = tf.cast(features['label'], tf.float32)

    return image, label


def get_tfrecord(num,images,labels):
    img = images
    label = labels
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)  #整个过程使用了两个线程
    return img_batch, label_batch

#定义预处理图像的函数
def pre_image(image_path):
    img = Image.open(image_path)
    img_1 = img.resize((299,299),Image.ANTIALIAS)
    img_arr = np.array(img_1)
    img_arr_1 = tf.reshape(img_arr, [1,299,299,3])
    return img_arr_1

def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()