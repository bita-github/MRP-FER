#!/usr/bin/env python
# coding: utf-8

import tensorflow 
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D,\
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization,\
    AveragePooling2D, Reshape, Permute, multiply
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_vggface import utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
from keras_vggface.vggface import VGGFace
import tensorflow
from PIL import Image
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.utils import class_weight
import os 
import random
import copy
from keras import Model
from keras.regularizers import l1, l2, l1_l2
import tensorflow
import skimage.transform
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential, model_from_json, load_model 
from keras.optimizers.schedules import InverseTimeDecay
import keras

plt.style.use('ggplot')


#batch generator reference: https://keras.io/preprocessing/image/

def my_generator(gen_args, b_size, im_size):
    
        """
            Function to create data bach generators which balance and perform data augmentation on each batch.
            Parameters: 
                gen_args: arguments of generator, dictionary 
                b_size: ssize of each batch, scalar
                im_size: image dimension (width, height), tuple
            Returns:
                train_it: train batch generator
                val_it: validation batch generator
                test_it: test batch generator
                class_weights_dict: estimated class weights for unbalanced datasets
        """    

        data_aug_gen = ImageDataGenerator(**gen_args)
        train_it = data_aug_gen.flow_from_directory('/home/ubuntu/Notebooks/Datasets/RAF_occ/train/', class_mode='categorical',
                                    batch_size=b_size, target_size=im_size)
        val_it = data_aug_gen.flow_from_directory('/home/ubuntu/Notebooks/Datasets/RAF_occ/validation/', class_mode='categorical',
                                    batch_size=b_size, target_size=im_size)
        

        classes = np.unique(trn_lbls['target'])
        class_weights = class_weight.compute_class_weight('balanced',classes, trn_lbls['target'])
        class_weights_dict = dict(enumerate(class_weights))
    
        return train_it, val_it, class_weights_dict


trn_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/RAF_occ/train_label.csv')

# Whether to retrain the model or load a previously saved model.
retrain = True
cp_dir ='/home/ubuntu/Notebooks/Models/FER_RAF_best.h5'
log_dir = '/home/ubuntu/Notebooks/Models/FER_RAF_log.csv'

if retrain:    
    VGGFace16_classes = 2622
    VGGFace16 = Sequential()

    #Block 1
    VGGFace16.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),
                         padding='same', activation='relu', name='conv1_1'))  
    VGGFace16.add(layers.Conv2D(filters=64,kernel_size=(3,3),
                         padding='same', activation='relu', name='conv1_2'))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    VGGFace16.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv2_1'))
    VGGFace16.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv2_2'))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
    VGGFace16.add(layers.Dropout(0.25))

    # Block 3
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_1'))
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_2'))
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_3'))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))
    VGGFace16.add(layers.Dropout(0.25))

    # Block 4
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_1'))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_2'))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_3'))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))
    VGGFace16.add(layers.Dropout(0.25))

    # Block 5
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_1'))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_2'))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_3'))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5'))
    VGGFace16.add(layers.Dropout(0.25))

    # Classification block
    VGGFace16.add(Flatten(name='flatten'))
    VGGFace16.add(layers.Dense(units=4096, name='fc6'))
    VGGFace16.add(Activation(activation='relu', name='fc6/relu'))
    VGGFace16.add(layers.Dense(units=4096, name='fc7'))
    VGGFace16.add(Activation(activation='relu', name='fc7/relu'))
    VGGFace16.add(layers.Dense(VGGFace16_classes, name='fc8'))
    VGGFace16.add(Activation(activation='softmax', name='fc8/softmax'))
    
    # load the pretrained weights
    VGGFace16.load_weights('/home/ubuntu/Notebooks/Models/vggface_vgg16.h5', by_name=True)
    
    # construct new classification layers
    nb_class = 7
    hidden_dim = 512
    initializer = tensorflow.keras.initializers.RandomNormal(mean=0., stddev=0.1)
    last_layer = VGGFace16.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=initializer, name='fc6')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=initializer, name='fc7')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_class, activation='softmax', kernel_initializer=initializer, name='fc8')(x)
    FER_VGGFace16 = tensorflow.keras.Model(VGGFace16.input, out )
    
    # compile the model
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    FER_VGGFace16.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    FER_VGGFace16.summary()
else:
    best_FER_VGGFace16 = load_model(cp_dir)
    best_FER_VGGFace16.summary()

# Fitting the model on the custom generator.
if retrain:
    # Defining paramters for image augmentation:
    generator_args = {'rescale' : 1./255,
                      'horizontal_flip':True}
    train_iter, val_iter, weights = my_generator(generator_args, b_size=64)
    # Save model logs and the best model to a file.
    model_log = CSVLogger(log_dir, separator=',') 
    model_cp = tensorflow.keras.callbacks.ModelCheckpoint(cp_dir, save_best_only=True,
                               monitor='val_loss', mode='min')
    
    # define early stop and learning decay
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

    # Fit the model.
    history = FER_VGGFace16.fit(x=train_iter, 
                  validation_data=(val_iter),
                  epochs=100,
                  class_weight=weights,
                  callbacks=[model_cp, model_log])
    best_FER_VGGFace16 = load_model(cp_dir)  # Retrieve the best model.

out = pd.read_csv(log_dir, delimiter=',')
plot_accuracy(out)
plot_loss(out)

FER = load_model('../Models/FER_RAF_best')

test_datagen = ImageDataGenerator(rescale=1./255)
test_iter= test_datagen.flow_from_directory(
        '../Datasets/RAF_occ_grayscale/validation/',
        target_size=(224, 224),
        shuffle = False,
        class_mode=None,
        batch_size=1)
test_iter.reset()

filenames = test_iter.filenames
nb_samples = len(filenames)
predict = FER.predict(test_iter, steps = nb_samples, verbose=1)

predicted_class_indices=np.argmax(predict,axis=1)
labels_ = (test_iter.class_indices)
labels_ = dict((v,k) for k,v in labels_.items())
predictions = [labels_[k] for k in predicted_class_indices]
y_preds = predicted_class_indices
y_test = test_iter.labels
filenames=test_iter.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

labels = {'anger': 0, 'disgust': 1,'fear': 2,
          'happiness': 3, 'neutral': 4, 'sadness': 5,
          'surprise': 6}
labels = list(labels.values())

accuracy_score(test_iter.labels, predicted_class_indices)
plot_precision_recall(list(y_test), list(y_preds))
plot_wrong_predictions_heatmap(list(y_test), list(y_preds))
cf_matrix = confusion_matrix(y_test, y_preds)
print(cf_matrix)
fig, ax = plt.subplots(figsize=(10,5))      
sns.heatmap(cf_matrix, annot=True, cmap='Blues', linewidths=10, fmt='g', ax=ax);

