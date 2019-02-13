import tensorflow as tf
from PIL import Image
import application
import detection
from labels import label

#将网上的模型和自己训练的模型结合在一起来预测
def prediction(img_path):
    image_path = img_path
    img = Image.open(image_path)
    w, h = img.size

    detections = detection.detection(image_path)
    class_index, prob = application.application(image_path)

    if detections:
        for each in detections:
            bxpts = each["box_points"]
            name = each["name"]
            if abs((w - bxpts[2]) - bxpts[0]) < 0.1 * w:
                print("有一个", name, "在你正前方")
            if w - bxpts[2] - bxpts[0] > 0:
                print("有一个", name, "在你左前方")
            if w - bxpts[2] - bxpts[0] < 0:
                print("有一个", name, "在你右前方")

    if prob > 0.98:
        print("在你前方有个", label[class_index])



