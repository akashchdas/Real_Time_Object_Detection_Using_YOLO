import os
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import struct
import cv2
from numpy import expand_dims
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.models import load_model, Model
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from NewBox import *
from BoundBox import *

model = load_model('YOLOmodel.h5')
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
input_h, input_w = 416, 416
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    image_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ret = cap.set(3,416)
    ret = cap.set(4,416)
    image = img_to_array(frame)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    yhat = model.predict(image)
    print([a.shape for a in yhat])
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
	    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    do_nms(boxes, 0.5)
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    for i in range(len(v_boxes)):
	    print(v_labels[i], v_scores[i])
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        height = y2 - y1
        #declaring a shift variable because due to poor quality video cam, the boxes are very big compared to object
        shift_x = (x2-x1)//8
        #changing y1 and y2 to top left and bottom right corner by adding/subtracting height
        cv2.rectangle(frame, (x1+shift_x,y1+height), (x2-shift_x,y2-height), (0,255,0), 4)
        text = "{}".format(v_labels[i])
        cv2.putText(frame, text, (x1,(y1+y2//2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow('face', frame)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()


