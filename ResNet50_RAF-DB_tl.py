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

def my_generator(gen_args, b_size=64, im_size = (224,224)):
    
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
    
        return train_it, val_it, test_it, class_weights_dict

trn_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/RAF_occ/train_label.csv')

# Whether to retrain the model or load a previously saved model.
retrain = True
cp_dir = '/home/ubuntu/Notebooks/Models/RAF_Resnet_best.h5'
log_dir = '/home/ubuntu/Notebooks/Models/RAF_Resnet_best_log.csv'


if retrain:    
  nb_class = 7
  vgg_model = VGGFace(model='resnet50', include_top=True, input_shape=(224, 224, 3))
  last_layer = vgg_model.get_layer('avg_pool').output
  x = keras.layers.Flatten(name='flatten')(last_layer)
  out = Dense(nb_class, activation='softmax', name='classifier')(x)
  FER_VGGFaceResnet50 = Model(vgg_model.input, out)
    
  # compile the model
  opt = tensorflow.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
  FER_VGGFaceResnet50.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
  FER_VGGFaceResnet50.summary()
else:
  best_FER_VGGFaceResnet50 = load_model(cp_dir)
  best_FER_VGGFaceResnet50.summary()

# Fitting the model on the custom generator.
if retrain:
    # Defining paramters for image augmentation:
    generator_args = {'rescale' : 1./255,
                      'horizontal_flip':True}
    train_iter, val_iter, weights = my_generator(generator_args, b_size=64)
    # Save model logs and the best model to a file.
    model_log = CSVLogger(log_dir, separator=',') 
    model_cp = keras.callbacks.ModelCheckpoint(cp_dir, save_best_only=True,
                               monitor='val_loss', mode='min')
    
    # define early stop and learning decay
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    
    # Fit the model.
    history = FER_VGGFaceResnet50.fit(x=train_iter, 
                  validation_data=(val_iter),
                  epochs=100,
                  class_weight=weights,
                  callbacks=[model_cp, model_log, rlrop, stop])
    best_FER_VGGFaceResnet50 = load_model(cp_dir)  # Retrieve the best model.

out = pd.read_csv(log_dir, delimiter=',')
plot_accuracy(out)
plot_loss(out)

FER = load_model('RAF_Resnet_best.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_iter= test_datagen.flow_from_directory(
        '../data/RAF_occ_grayscale/validation/',
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

