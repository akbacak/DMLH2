#coding=utf-8

import cv2
import glob
import numpy as np
import sys,os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import glob
import os, os.path
import re
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Sequential,Input,Model,InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.applications import VGG16
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras.preprocessing.image import ImageDataGenerator



Y=scipy.io.loadmat('/home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets')
del Y['__version__']
del Y['__header__']
del Y['__globals__']
Y = list(Y.values())
Y = np.reshape(Y, (2000,5))


image_size=224
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


X = np.load(open('preprocessed_X.npy'))
X.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1 ,random_state=43)


batch_size = 32
epochs = 120
hash_bits = 512

#visible = Input(shape=(7,7,512)) 
visible = Input(shape = base_model.output_shape[1:])
Flatten = Flatten()(visible)
Dense_1 = Dense(4096)(Flatten)
Dense_2 = Dense(hash_bits ,activation='sigmoid')(Dense_1)
Dense_3 = Dense(5, activation='sigmoid')(Dense_2)
model = Model(input = visible, output=Dense_3)
print(model.summary())

# https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance

import keras.backend as K
# e = 0.5
def c_loss(noise_1, noise_2):
    def loss(y_true, y_pred):
        return (K.binary_crossentropy(y_true, y_pred) + (K.sum((noise_1 - noise_2)**2) ) * (1/hash_bits)  )
 
    return loss



from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = c_loss(noise_1 = tf.to_float(Dense_2 > 0.5 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy'])
history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )

model_json = model.to_json()
with open("models/dmlh2_512_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/dmlh2_512_weights.h5")



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



'''
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

