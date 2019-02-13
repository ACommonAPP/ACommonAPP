from imageai.Detection import ObjectDetection
import os
import time

# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"])
#     print("--------------------------------")
#
# print ("\ncost time:",end-start)

def detection(image_path):

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()

    # 载入已训练好的文件
    detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(car=True)
    # 将检测后的结果保存为新图片
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path, image_path),
                                                 output_image_path=os.path.join(execution_path,
                                                                                image_path.split(sep='.')[0]+"_new.jpg"),
                                                       minimum_percentage_probability=70)

    return detections