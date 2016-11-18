###
# prediction for self training 
#
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adadelta
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.utils import np_utils
import cPickle
from random import shuffle
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf
import sys
K.set_image_dim_ordering('th')

pickle_dir = sys.argv[1]
model_name = sys.argv[2]
output = sys.argv[3]
os.chdir(pickle_dir)

tst = cPickle.load(open('test.p','rb'))
tst_data = np.array(tst['data'])
tst_data = tst_data.reshape(-1,3,32,32)
tst_data = tst_data.astype('float32')
tst_data /= 255

model = load_model(model_name)

predict_y = model.predict_classes(tst_data)
ans = np.empty((predict_y.shape[0],2))
for i in range(predict_y.shape[0]):
    ans[i] = np.array([i,predict_y[i]])
np.savetxt(output, ans, delimiter = ',',fmt = "%i", header='ID,class', comments = '')
