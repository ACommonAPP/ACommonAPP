import os
import time
from prediction import prediction

while True:
    filenames = os.listdir("./image")
    if filenames:
        for filename in filenames:
            prediction("./image/"+filename)
        for filename in filenames:
            os.remove("./image/"+filename)
            os.remove("./image/"+filename.split(sep=".")[0]+"_new.jpg")
    time.sleep(100)

