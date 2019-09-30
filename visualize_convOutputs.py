import cv2
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import Model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras import models
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


model   = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))
model.summary()

img_path = '/home/ubuntu/caffe/data/lamda_2/lamdaPics/562.jpg'
layer_outputs = [layer.output for layer in model.layers[:20]]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
img = image.load_img(img_path, target_size = (224,224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /=255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:20]]
layer_outputs = layer_outputs[1:] # SEE https://github.com/keras-team/keras/issues/10372
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print (first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis') # Fourth channel activation of the first layer
plt.show()

plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis') # Seventh channel activation of the first layer
plt.show()





layer_names = []
for layer in model.layers[:20]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            np.seterr(divide='ignore', invalid='ignore') # SEE https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()












'''
img = plt.imread("/home/ubuntu/caffe/data/lamda_2/lamdaPics/562.jpg")
img = np.array(img)    # converting list to array
img = resize(img, preserve_range=True, output_shape=(224,224,3)).astype(int)      # reshaping to 224*224*3
img = np.array(img)
img = preprocess_input(img, mode='tf')      # preprocessing the input data
img = np.reshape(img, (1, 224, 224, 3))
model   = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))
print(model.summary())


for layer in model.layers[:5]:
    print(layer.name)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
intermediate_output = intermediate_layer_model.predict(img)
plt.matshow(intermediate_output[0, :, :, 12], cmap='viridis')
plt.show()
'''
