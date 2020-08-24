#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os 
import random
import copy
import tensorflow 
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D,\
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization,\
    AveragePooling2D, Reshape, Permute, multiply
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
import tensorflow
from PIL import Image
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.utils import class_weight
from keras import Model
import skimage.transform
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential, model_from_json, load_model 
from keras.regularizers import l1, l2, l1_l2
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

plt.style.use('ggplot')

#functions for visualization of the results

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
        Function to plot the precision and recall of a model per class
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
        Function to plot a heatmap of the number of wrong prediction per pair of classes
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

trn_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/train_label.csv')


# Whether to retrain the model or load a previously saved model
retrain = False
cp_dir = '/home/ubuntu/Notebooks/Models/FER_VGGFace16.h5'
log_dir = '/home/ubuntu/Notebooks/Models/FER_VGGFace16_log.csv'


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
                         padding='same', name='conv2_1',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv2_2',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_1',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_2',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv3_3',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_1',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_2',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv4_3',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_1',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_2',kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='conv5_3',  kernel_regularizer=l2(5*1e-4)))
    VGGFace16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5'))

    # Classification block
    VGGFace16.add(Flatten(name='flatten'))
    VGGFace16.add(layers.Dense(units=4096, name='fc6'))
    VGGFace16.add(Activation(activation='relu', name='fc6/relu'))
    VGGFace16.add(layers.Dense(units=4096, name='fc7'))
    VGGFace16.add(Activation(activation='relu', name='fc7/relu'))
    VGGFace16.add(layers.Dense(VGGFace16_classes, name='fc8'))
    VGGFace16.add(Activation(activation='softmax', name='fc8/softmax'))
    
    # load the pretrained weights
    VGGFace16.load_weights('/home/ubuntu/Notebooks/Models/vggface_vgg16.h5')
    
    # construct new classification layers
    nb_class = 8
    hidden_dim = 512
    last_layer = VGGFace16.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = layers.Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = layers.Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = layers.Dense(nb_class, activation='softmax', name='fc8')(x)
    FER_VGGFace16 = tensorflow.keras.Model(VGGFace16.input, out )
    
    # compile the model
    opt = tensorflow.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    FER_VGGFace16.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    FER_VGGFace16.summary()
else:
    best_FER_VGGFace16 = load_model(cp_dir)
    best_FER_VGGFace16.summary()


# Fitting the model on the custom generator
if retrain:
    # Defining paramters for image augmentation
    generator_args = {'rescale' : 1./255,
                      'horizontal_flip':True}
    train_iter, val_iter, test_iter, weights = my_generator(generator_args, b_size=64)
    # Save model logs and the best model to a file
    model_log = CSVLogger(log_dir, separator=',') 
    model_cp = ModelCheckpoint(cp_dir, save_best_only=True,
                               monitor='val_loss', mode='min')

    # define learning rate decay and early stop 
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, cooldown=5)
    stop = EarlyStopping(monitor='loss', patience=10)
   
    # Fit the model
    history = FER_VGGFace16.fit(x=train_iter, 
                  validation_data=(val_iter),
                  epochs=50,
                  class_weight=weights,
                  callbacks=[model_cp, model_log, rlrop, stop])
    best_FER_VGGFace16 = load_model(cp_dir)  # Retrieve the best model.



out = pd.read_csv(log_dir, delimiter=',')
plot_accuracy(out)
plot_loss(out)


#create test generator 
test_datagen = ImageDataGenerator(rescale=1./255)
test_iter= test_datagen.flow_from_directory(
        '../data/FERPlus_occ/test/',
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
           'fear': 3, 'happiness': 4, 'neutral':5, 'sadness': 6, 'surprise': 7}
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
sns.heatmap(cf_matrix, annot=True, cmap='Blues', linewidths=10, fmt='g', ax=ax); 



