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
    time.sleep(100)

