from __future__ import print_function
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
import os.path
import tensorflow as tf
from keras import backend as K
import sys
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
tf.python.control_flow_ops = tf
import os
os.chdir(sys.argv[1])
output_model = sys.argv[2]
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

#LAB MACHINE CONFIG
miulab_gpu = False
if miulab_gpu == True:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.20
    sess = tf.Session(config = config)
    K.set_session(sess)



#labeled data
x = cPickle.load(open('all_label.p','rb'))
X = np.array(x).reshape(-1,3,32,32) 

Y = np.empty(0)
for i in range(10):
    Y = np.append(Y,np.zeros(500)+(i))
ind = np.arange(X.shape[0])
validation_num = 80
shuffle(ind)
X = X[ind]
Y = Y[ind]
X_train = X[:-validation_num] 
y_train = Y[:-validation_num]
X_test = X[-validation_num:] 
y_test = Y[-validation_num:]

#params
batch_size = 16
nb_classes = 10
nb_epoch = 60
data_augmentation = True
pre_train = True
self_only = False
full_train = pre_train and not self_only
threshold = 0.999
self_training = 10

#unlabeled data
u = cPickle.load(open('all_unlabel.p','rb'))
U = np.array(u).reshape(-1,3,32,32)

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
#4
elu = ELU(alpha=1.0)

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X_train.shape[1:], W_constraint=maxnorm(3)))
model.add(elu)
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering="th"))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same' ))
model.add(elu)
model.add(Convolution2D(128, 3, 3, border_mode='same', W_constraint=maxnorm(3)))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering="th"))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 3, 3, border_mode='same' ))
model.add(elu)
model.add(Convolution2D(256, 3, 3, border_mode='same', W_constraint=maxnorm(3)))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering="th"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print (model.summary())

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
U = U.astype('float32')
X_train /= 255
X_test /= 255
U /= 255

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
if pre_train :
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
            # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_test, Y_test),
                                callbacks=[early_stopping])
        scores = model.evaluate(X_test, Y_test, verbose=0)
        model.save('model'+str(scores[1])+'.h5')
if full_train or self_only :
    print ('perform self-training')
    # if self_only:
    #     del model
    #     model = load_model('md0760.h5')     
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    for i in range(self_training):       
        print ('round',i)
        if U.shape[0] < 100:
            print ('pass')
            pass
        shuffle(U)
        prob = model.predict_proba(U)
        confident = np.where(prob.max(axis = 1) > threshold)
        not_confident = np.delete(np.arange(len(prob)), np.array(confident))
        print (confident[0].shape, not_confident.shape)
        print ('adding ',len(confident[0]),'samples')
        
        X_new = U[confident]
        Y_new = prob[confident].argmax(axis = 1)
        U = U[not_confident]
        X_train = np.append(X_train,X_new,axis = 0)
        Y_train = np.append(Y_train,np_utils.to_categorical(Y_new, nb_classes),axis = 0)
        #import pdb; pdb.set_trace()
        print(X_train.shape[0], 'train samples')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=32),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=20,
                            validation_data=(X_test, Y_test),
                            callbacks=[early_stopping])
    model.save(output_model)

