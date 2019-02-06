import cv2
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images




image_1 = plt.imread("/home/ubuntu/caffe/data/lamda_2/lamdaPics/1995.jpg") 
image_1 = np.array(image_1)    # converting list to array

image_1 = resize(image_1, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_1 = np.array(image_1)


from keras.applications.vgg16 import preprocess_input
image_1 = preprocess_input(image_1, mode='tf')      # preprocessing the input data
image_1 = np.reshape(image_1, (1, 224, 224, 3))

image_size=224
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

batch_size=64
image_1 = base_model.predict(image_1, batch_size=batch_size, verbose=0, steps=None)



#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('models/dmlh2_64_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("models/dmlh2_64_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)


features = model.predict(image_1, batch_size=64, verbose=0, steps=None)[0] #If we  do not use [0], output will be [[ ... ]] 
features = features > 0.5
features = features.astype(int)
print(features)

