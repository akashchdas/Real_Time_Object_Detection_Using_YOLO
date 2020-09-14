from model.WeightReader import *
from model.ModelStructure import *

model = make_yolov3_model()
weights = WeightReader('yolov3.weights')
weights.load_weights(model)
 
model.save('YOLOmodel.h5')

