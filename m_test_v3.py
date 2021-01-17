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





hash_bits = 16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable =False
inputs  = Input(shape=(image_size, image_size, 3))
enver       = base_model(inputs, training=False)
Flatten = Flatten()(enver)
Dense_1 = Dense(4096)(Flatten)
Dense_2 = Dense(hash_bits ,activation='sigmoid')(Dense_1)
output1 = Dense(1, activation = 'sigmoid')(Dense_2)
output2 = Dense(1, activation = 'sigmoid')(Dense_2)
output3 = Dense(1, activation = 'sigmoid')(Dense_2)
output4 = Dense(1, activation = 'sigmoid')(Dense_2)
output5 = Dense(1, activation = 'sigmoid')(Dense_2)
model = Model(inputs, [output1,output2,output3,output4,output5])
print(model.summary())



import keras.backend as K
from tensorflow.keras.optimizers import SGD
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer = sgd, loss="binary_crossentropy")


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(5)])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN)
validation_data=generator_wrapper((valid_generator),validation_steps=STEP_SIZE_VALID,epochs=1,verbose=2)


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)





