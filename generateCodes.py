#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('dmlh_v2_aug_64_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("dmlh_v2_aug_64_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

X = np.load(open('preprocessed_X.npy'))
X.shape

XX = model.predict(X, batch_size=32, verbose=0, steps=None) #     TRY THIS ALSO XX = model.predict(X, batch_size)

features = XX > 0.5
features = features.astype(int)
np.savetxt('dmlh_v2_new_aug_64.txt',features, fmt='%d')


