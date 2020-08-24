#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow 
import os 
import random
import copy
from keras.preprocessing.image import ImageDataGenerator 
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
import keras_vggface
from keras_vggface.vggface import VGGFace
import tensorflow
from PIL import Image
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.utils import class_weight
from keras import Model
import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential, model_from_json, load_model 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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
        train_it = data_aug_gen.flow_from_directory('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/train/', class_mode='categorical',
                                    batch_size=b_size, target_size=im_size)
        val_it = data_aug_gen.flow_from_directory('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/validation/', class_mode='categorical',
                                    batch_size=b_size, target_size=im_size)
        test_it = data_aug_gen.flow_from_directory('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/test/', class_mode='categorical',
                                              batch_size=b_size, target_size=im_size)
        

        classes = np.unique(trn_lbls['target'])
        class_weights = class_weight.compute_class_weight('balanced',classes, trn_lbls['target'])
        class_weights_dict = dict(enumerate(class_weights))
    
        return train_it, val_it, test_it, class_weights_dict


#functions for visualization of the results.

def plot_accuracy(model_out):
    """
        Function to plot the accuracy vs. epoch of a model
        Parameters:
            model_out: the output of the fit function of the model
    """
    trn_accs = model_out['accuracy']
    val_accs = model_out['val_accuracy']
    epochs = np.arange(len(trn_accs))

    plt.figure(figsize=(15, 5))
    plt.plot(trn_accs)
    plt.plot(val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.xticks(ticks=epochs, labels=epochs + 1)
    plt.legend(['Training', 'Validation'])

    plt.show()

def plot_loss(model_out):
    """
        Function to plot the loss vs. epoch of a model given the output of its fit 
        Parameters:
            model_out: the output of the fit function of the model
    """
    trn_accs = model_out['loss']
    val_accs = model_out['val_loss']
    epochs = np.arange(len(trn_accs))

    plt.figure(figsize=(15, 5))
    plt.plot(trn_accs)
    plt.plot(val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.xticks(ticks=epochs, labels=epochs + 1)
    plt.legend(['Training', 'Validation'])

    plt.show()

def plot_precision_recall(targets, preds):
    """
        Function to plot the precision and recall of a model per class.
        Parameters:
            targets: the target values, list
            preds: the predicted values, list
    """
    acc = accuracy_score(targets, preds)
    print('Accuracy:', acc)

    # Check whether we are predicting class labels or indices.
    if preds[0] in labels:
        labs = [i for i in labels if i in preds or i in targets]
        x = [labels.index(i) for i in labs]
    else:
        labs = [labels.index(i) for i in labels if 
                labels.index(i) in preds or labels.index(i) in targets]
        x = labs

    recalls = recall_score(targets, preds, average=None, labels=labs)
    precisions = precision_score(targets, preds, average=None, labels=labs)

    df = pd.DataFrame(np.c_[x, recalls, precisions], 
                    columns=['labels', 'Recalls', 'Precisions'])
    df['labels'] = df['labels'].astype(int)
    df = df.melt(id_vars='labels')


    plt.figure(figsize=(10, 5))
    sb.barplot(x='labels', y='value', hue='variable', data=df)
    plt.legend(loc='lower right')
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("The precision and recall on each class")
    plt.show()

def plot_wrong_predictions_heatmap(targets, preds):
    """
        Function to plot a heatmap of the number of wrong prediction per pair of classes.
        Parameters:    
            targets: the target values, list
            preds: the predicted values, list
    """
    if preds[0] in labels:
        targets = [labels.index(i) for i in targets]
        preds = [labels.index(i) for i in preds]

    pred_table = pd.DataFrame(
        np.c_[targets, preds], columns=['Real', 'Prediction'])
    pred_table = pred_table[pred_table['Real'] != pred_table['Prediction']]
    hm_matrix = pd.crosstab(pred_table['Real'], pred_table['Prediction'])

    max_tick = hm_matrix.values.max() + 1
    ticks=np.arange(0, max_tick, np.ceil(max_tick / 10))
    boundaries = np.arange(-0.5, max_tick)
    cmap = plt.get_cmap("Reds", max_tick)
    
    plt.figure(figsize=(10, 5))
    ax = sb.heatmap(hm_matrix, annot=True, linewidths=0.01, 
                    cmap=cmap, linecolor='k',
                    cbar_kws={"ticks":ticks, "boundaries":boundaries})
    ax.set_title("Number of wrong predictions for each pair of target and " +
                 "predicted values")

    plt.tight_layout()
    plt.show()

trn_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/train_label.csv')

# Whether to retrain the model or load a previously saved model.
retrain = True
cp_dir = '/home/ubuntu/Notebooks/Models/FER_Resnet_v2_best.h5'
log_dir = '/home/ubuntu/Notebooks/Models/FER_Resnet_v2_log.csv'

if retrain:    
  nb_class = 8
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
    train_iter, val_iter, test_iter, weights = my_generator(generator_args, b_size=64)
    # Save model logs and the best model to a file.
    model_log = CSVLogger(log_dir, separator=',') 
    model_cp = keras.callbacks.ModelCheckpoint(cp_dir, save_best_only=True,
                               monitor='val_loss', mode='min')

    # Fit the model.
    history = FER_VGGFaceResnet50.fit(x=train_iter, 
                  validation_data=(val_iter),
                  epochs=50,
                  class_weight=weights,
                  callbacks=[model_cp, model_log])
    best_FER_VGGFaceResnet50 = load_model(cp_dir)  # Retrieve the best model.

# plot accuracy and loss
    out = pd.read_csv(log_dir, delimiter=',')
plot_accuracy(out)
plot_loss(out)

# load best model
FER = load_model('FER_Resnet_v2_best.h5')
#create test generator 
test_datagen = ImageDataGenerator(rescale=1./255)
test_iter= test_datagen.flow_from_directory(
        '../Datasets/FERPlus_occ/test/',
        target_size=(224, 224),
        shuffle = False,
        class_mode=None,
        batch_size=1)
test_iter.reset()

#retrieve image filenames generated by test batch generator
filenames = test_iter.filenames
nb_samples = len(filenames)

#prediction
predict = best_FER_VGGFace16.predict(test_iter, steps = nb_samples, verbose=1)

# create a dataframe for images and their corresponding predicted target 
predicted_class_indices=np.argmax(predict,axis=1)
labels_ = (test_iter.class_indices)
labels_ = dict((v,k) for k,v in labels_.items())
predictions = [labels_[k] for k in predicted_class_indices]
y_preds = predicted_class_indices
y_test = test_iter.labels
filenames=test_iter.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

labels = {'anger': 0, 'contempt': 1, 'disgust':2,
          'fear': 3, 'happiness': 4, 'neutral':5,
          'sadness': 6, 'surprise': 7}
labels = list(labels.values())

#evaluation metrics
acc = accuracy_score(test_iter.labels, predicted_class_indices)
print(f'Accuracy score:{acc}')
f1 = f1_score(y_test, y_preds, average=None, labels=labels)
print(f'F1 score:{f1}')
cf_matrix = confusion_matrix(y_test, y_preds)
plot_precision_recall(list(y_test), list(y_preds))
plot_wrong_predictions_heatmap(list(y_test), list(y_preds))
fig, ax = plt.subplots(figsize=(10,5))        
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues', linewidths=10, ax=ax)
plt.show()

