from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG16
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import pandas as pd
import numpy as np
import tensorflow as tf



df=pd.read_csv('./miml_dataset/miml_labels_1.csv')

columns=["desert", "mountains", "sea", "sunset", "trees"]

datagen=ImageDataGenerator(rescale=1./255.)

test_datagen=ImageDataGenerator(rescale=1./255.)

image_size = 224

train_generator=datagen.flow_from_dataframe(
dataframe=df[:1800],
directory="./miml_dataset/images",
x_col="Filenames",
y_col=columns,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(image_size,image_size))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=df[1800:1900],
directory="./miml_dataset/images",
x_col="Filenames",
y_col=columns,
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(image_size,image_size))

test_generator=test_datagen.flow_from_dataframe(
dataframe=df[1900:],
directory="./miml_dataset/images",
x_col="Filenames",
batch_size=1,
seed=42,
shuffle=False,
class_mode = None,
target_size=(image_size,image_size))





hash_bits = 32

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable =False
inputs  = Input(shape=(image_size, image_size, 3))
enver       = base_model(inputs, training=False)
Flatten = Flatten()(enver)
Dense_1 = Dense(2048)(Flatten)
Dense_2 = Dense(hash_bits ,activation='sigmoid')(Dense_1)
Dense_3 = Dense(5, activation='sigmoid')(Dense_2)
model   = Model(inputs, Dense_3)
print(model.summary())

'''
from tensorflow.keras.optimizers import SGD
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss="binary_crossentropy")
'''

def loss_1(y_true, y_pred):
    return  (tf.keras.losses.binary_crossentropy(y_true, y_pred))

def loss_2(noise_1, noise_2 ):
    return  (tf.keras.losses.binary_crossentropy(noise_1, noise_2))

import keras.backend as K
def loss(loss_1, loss_2):
    return  K.mean(loss_1) +  K.mean(loss_2)


sgd =optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = loss_1,  optimizer=sgd, metrics=['accuracy'])
#model.compile(loss =  loss_1 + loss_2(noise_1 = tf.cast(Dense_2 > 0.5, tf.float32 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy'])



'''
# This is working
def c_loss(y_true, y_pred):
    return (tf.keras.losses.binary_crossentropy(y_true, y_pred))

sgd =optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = c_loss,  optimizer=sgd, metrics=['accuracy'])
'''



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=9
)


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

pred_bool = (pred >0.5)

predictions = pred_bool.astype(int)
columns=["desert", "mountains", "sea", "sunset", "trees"]
#columns should be the same order of y_col

results=pd.DataFrame(predictions, columns=columns)
results["Filenames"]=test_generator.filenames
ordered_cols=["Filenames"]+columns
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
