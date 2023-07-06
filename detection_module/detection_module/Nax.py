import numpy as np
import cv2
import os.path as op
from glob import glob

image_path = "/home/gorilla/220311_1920_seoul_0014/image"
image_file = op.join(image_path, "000045.jpg")

image = cv2.imread(image_file)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
result,encimg =cv2.imencode('.jpg', image, encode_param)

#decode from jpeg format
decimg=cv2.imdecode(encimg,1)


cv2.imshow("image", decimg)
cv2.waitKey()
