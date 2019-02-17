  在这inception_v3里的几个文件中，foward.py,backward.py,generateds是用来训练的,test.py是用来测试的,application.py是用来应用的。detection是imageai的接口，application是我训练的模型的接口，prediction是总接口，将detection和prediction结合起来,keep_pred实现持续地预测图片，前提是要有外部每隔一段时间地传入图片。inception_v3中本来有个image文件夹（不知为何没有了），是用来接受外部传入地图片地文件夹，一旦有图片传入，keep_pred将读取图片，并预测，返回简略的提示结果，然后删除图片，每隔100ms检查有无图片。  训练的原理：有个data文件夹，只要将图片放入data/train/images中，然后将标签地txt文件写好，放在data/train/labels.txt中（标签格式：图片名称 类别下标。只要将图片名称设为2(100).jpg的形式，2为类别下标，然后运行位于data/train/的write.py文件，就可以了），然后运行backward.py就可以训练了
  
  



