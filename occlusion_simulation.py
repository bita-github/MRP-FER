#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os 
import random
import copy
from keras import Model, layers
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OneHotEncoder
import skimage.transform
import cv2
import dlib 
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner, rect_to_bb
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential, model_from_json 
from PIL import Image

import geopandas as gpd
import plotly.graph_objects as go

def load_toarray(lbls, dset):
    '''
        Function to load images of each set into numpy arrays
        Parameters: 
          lbls: dataframe 
          dset: string ('training', 'validation', 'test')
        Returns:
            ndarray (number of images, image width in pixel , image height in pixel)
    '''
    base_path = '/home/ubuntu/Notebooks/Datasets/FERPlus/'
    X = []
    for i in range(len(lbls)):
        img = Image.open(os.path.join(base_path, dset, lbls['filename'][i]))
        X.append(np.asarray(img))
    return np.array(X)


def majority_vote(labels, emotions = {'neutral': 0, 'happiness': 1, 'surprise': 2,
           'sadness': 3, 'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7}):
    '''
        Function to pick the target variable based on majority vote
        Parameters: 
          labels: dataframe 
          emotions: dictionary ('class name': class code)
        Returns:
            target: list with shape of (number of instances,)
    '''
    
    target = []

    for i in range(len(labels)):
        m_vote = 0
        m_idx = 0
        for emo in emotions.keys():
            if labels[emo][i] > m_vote:
                m_vote = labels[emo][i]
                m_idx = emotions[emo]
            elif labels[emo][i] == m_vote :
                m_idx = emotions[emo]
        target.append(m_idx)
    
    return target 


def plot_class_distribution(y_, dataset):
  '''
    Function to interactivly visualize the distribution of the number of samples per class
    Parameters: 
      y_: ndarray with shape (number of instances) : target variable
      dataset: string ('training', 'validation', 'public test', 'private test')
  '''
  y_values = sorted(np.unique(y_))
  bars = [np.sum(y_ == yval) for yval in y_values]
  title = f'The distribution of number of samples per class in {dataset} set'
  fig = go.Figure(data=[go.Bar(
    x=y_values,
    y=bars,
    text=list(emotions.keys()))])
  fig.update_layout(
                  width = 600, title={'text': title,
                                      'y':0.88,'x':0.48,'xanchor': 'center','yanchor': 'top'}, 
                  yaxis_title="Number of instance", font=dict(size=9),
                  xaxis = dict(tickmode = 'linear', title = 'Class'), template = 'seaborn'
                 )
  fig.show()


def face_detection(x):
    '''
        Function to detect faces in the image
          Parameters:
            x: ndarray with shape (number of images, image width in pixel, image height in pixel), grayscale 
          Returns:
            shapes: ndarray with shape (number of images, 68, 2), landmark coordinates
            missed: list, indexes of images where face detection failed
    '''
    len_data = len(x)
    shapes = list(np.zeros(len_data))
    rects = []
    missed = []
    #load the input image and convert it to grayscale if it has 3 channel
    for i in range(len_data):
      image = x[i]
      if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      else:
        gray = np.uint8(image)

      #detect faces in the image
      rects.append(detector(gray, 1))

      #extract landmarks from the image
      if len(rects[i]) >= 1:
        shapes[i] = landmark_detection(rects[i], gray)
      else:
        missed.append(i)
    return shapes, missed

def landmark_detection(recs, g_img):
  '''
    Function to detect face landmarks
      Parameters:
        recs: dlib.rectangles, detected faces in the image
        g_img: ndarray with shape (image width in pixel, image height in pixel), grayscale
      Returns:
        shape: ndarray with shape (68, 2), landmark coordinates
  '''
  for (i, rect) in enumerate(recs):
      # determine the facial landmarks for the detected face 
      shape = predictor(g_img, rect)
      # convert the facial landmark (x, y)-coordinates to an array
      shape = face_utils.shape_to_np(shape)
  return shape

def scale_VR(dim, shape_):
  '''
  Function to scale VR headset dimension based on the distance between 2 temporal bones
    Parameters:
      dim:list [w, h], width and height of the VR headset 
      shape: ndarray with shape (68, 2)
    return:
      w: integer, scaled width of the VR headset
      h: integer, scaled height of the VR headset
  '''
  ratio = dim[1]/dim[0]
  right_temple = shape_[0]
  left_temple = shape_[16]
  dY = right_temple[1] - left_temple[1]
  dX = right_temple[0] - left_temple[0]
  dist = np.sqrt(dY ** 2 + dX ** 2)
  w = int(dist)
  h = int(ratio * w)
  return w, h

def find_EyeCentre(shape):
  '''
  Function to determine centre of each detected eyes 
    Parameters:
      shape: ndarray with shape (68, 2)
    return:
      rcent: ndarray with shape (2,), right eye centre x and y coordinates
      lcent: ndarray with shape (2,), left eye centre x and y coordinates
  '''
  # extract the left and right eye x and y coordinates
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
  rightEyePts = shape[rStart:rEnd]
  leftEyePts = shape[lStart:lEnd]

  # compute the center of each eye
  rcent = rightEyePts.mean(axis=0).astype('int')
  lcent = leftEyePts.mean(axis=0).astype('int')

  return rcent, lcent

def get_VR_points(m, d):
  '''
  Function to compute 4 corner points of VR headset based on eye position 
    Parameters:
      m: tuple(x, y), coordinate of centre between two eyes
      d: list [w, h], width and height of VR headset
    return:
      ur_vr: tuple(x, y), coordinate of upper-right point in VR headset
      br_vr: tuple(x, y), coordinate of bottom-right point in VR headset
      bl_vr: tuple(x, y), coordinate of bottom-left point in VR headset
      ul_vr: tuple(x, y), coordinate of upper-left point in VR headset
  '''
  VR_w = d[0] // 2
  VR_h = d[1] // 2
  bl_vr = m[0] + VR_w, m[1] + VR_h
  ul_vr = m[0] + VR_w, m[1] - VR_h
  ur_vr = m[0] - VR_w, m[1] - VR_h
  br_vr = m[0] - VR_w, m[1] + VR_h
  
  return ur_vr, br_vr, bl_vr, ul_vr


def rotate_pts(rcent, lcent, pts, m):
  '''
  Function to rotate points around a midpoint
    Parameters:
      rcent: ndarray with shape (2,), right eye centre x and y coordinates
      lcent: ndarray with shape (2,), left eye centre x and y coordinates
      pts: tuple of tuples (4,2), position of VR headset corner points
      m: tuple(x, y), coordinate of centre between two eyes
    return:
      rotated: list of rotated points around a midpoint
  '''
  dY = rcent[1] - lcent[1]
  dX = rcent[0] - lcent[0]
  theta = np.arctan(dY/dX)

  rotation_matrix = np.array(( (np.cos(theta), -np.sin(theta)),
                               (np.sin(theta), np.cos(theta)) ))

  rotated=[]
  for i in range(len(pts)):
    v = np.array(pts[i]) - np.array(m)
    rotated.append(tuple(rotation_matrix.dot(v).astype(int) + m))
  
  return rotated


def VR_patch(x, shapes, VR_dim=[20, 10]):
      '''
          Function to apply VR patch  on image
            Parameters:
              x: ndarray with shape (number of images, image width in pixel, image height in pixel), grayscale
              shapes: ndarray with shape (number of images, 68, 2), landmark coordinates
              VR_dim: VR headset dimension [w, h] 
      '''
    len_data = len(x)
    for i in range(len_data):
        image = x[i]
        VR_dim_scaled = scale_VR(VR_dim, shapes[i])
        # compute VR headset position on the face
        # determine each eye's centre 
        rightEyeCenter, leftEyeCenter = find_EyeCentre(shapes[i])
        # find the centre between two eyes
        midpoint = (rightEyeCenter[0] + leftEyeCenter[0]) // 2, (rightEyeCenter[1] + leftEyeCenter[1]) // 2
        # compute VR points 
        VR_pts = get_VR_points(midpoint, VR_dim_scaled)
        # determine VR headset aligned with eyes' position
        VR_pts_rotated = rotate_pts(rightEyeCenter, leftEyeCenter, VR_pts, midpoint)
        # overlay VR headset patch on the image
        cv2.fillPoly(image, [np.int32(tuple(VR_pts_rotated))], 1, 255)


def dataset_structure(x, labels, dset, base_path):
    
    #create directory for each class
    for emo in emotions.keys():
        path = os.path.join(base_path, dset, emo) 
        if not (os.path.isdir(path)): os.mkdir(path) 
            
    #copy images to their respecticve emotion class
    for i in range(len(labels)):
        if labels['target'][i] == 0:   
            dst = os.path.join(base_path, dset, 'neutral/', labels['filename'][i])
        elif labels['target'][i] == 1:   
            dst = os.path.join(base_path, dset, 'happiness/', labels['filename'][i])
        elif labels['target'][i] == 2:   
            dst = os.path.join(base_path, dset, 'surprise/', labels['filename'][i])
        elif labels['target'][i] == 3:   
            dst = os.path.join(base_path, dset, 'sadness/', labels['filename'][i])
        elif labels['target'][i] == 4:   
            dst = os.path.join(base_path, dset, 'anger/', labels['filename'][i])
        elif labels['target'][i] == 5:   
            dst = os.path.join(base_path, dset, 'disgust/', labels['filename'][i])
        elif labels['target'][i] == 6:   
            dst = os.path.join(base_path, dset, 'fear/', labels['filename'][i])
        elif labels['target'][i] == 7:   
            dst = os.path.join(base_path, dset, 'contempt/', labels['filename'][i])
        (Image.fromarray(x[i])).save(dst)


# FERPlus dataset includes 8 different emotions 
# https://arxiv.org/abs/1608.01041 
#https://github.com/microsoft/FERPlus
emotions = {'Neutral': 0, 'Happiness': 1, 'Surprise': 2,
           'Sadness': 3, 'Anger': 4, 'Disgust': 5, 'Fear': 6, 'Contempt': 7}

#read target variables of train, test and validation sets
columns = ['filename', 'usage', 'neutral', 'happiness', 'surprise',
           'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']


trn_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/FERPlus/train/label.csv',
                       header=None, names=columns)
val_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/FERPlus/validation/label.csv',
                       header=None, names=columns)
test_lbls = pd.read_csv('/home/ubuntu/Notebooks/Datasets/FERPlus/test/label.csv',
                        header=None, names=columns)

trn_lbls['target'] = majority_vote(trn_lbls)
val_lbls['target'] = majority_vote(val_lbls)
test_lbls['target'] = majority_vote(test_lbls)

plot_class_distribution(trn_lbls['target'], 'train')
plot_class_distribution(val_lbls['target'], 'validation')
plot_class_distribution(test_lbls['target'], 'test')


X_trn = load_toarray(trn_lbls, 'train')
X_val = load_toarray(val_lbls, 'validation')
X_test = load_toarray(test_lbls, 'test')

set(trn_lbls['target'])

emotions = {'Neutral': 0, 'Happiness': 1, 'Surprise': 2,
           'Sadness': 3, 'Anger': 4, 'Disgust': 5, 'Fear': 6, 'Contempt': 7}

num_classes = len(set(trn_lbls['target']))
size = (9, 4)
fig, axs = plt.subplots(*size, figsize=(10, 18))
axs = axs.flatten()
for i in range(num_classes):
    for j in range(len(trn_lbls)):
        if trn_lbls['target'][j] == i: 
            axs[i].imshow(X_trn[j], cmap='gray')
            axs[i].set_title(list(emotions.keys())[i])
            axs[i].axis('off')
            break
for i in range(num_classes, size[0] * size[1]):
    axs[i].remove()
plt.show() 

#initialize dlib’s pretrained face detector based on a modification to the standard Histogram of
#Oriented Gradients + Linear SVM method for object detection.
shape_predictor = '/home/ubuntu/Notebooks/Models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

#detect faces and landmark in faces 
X_trn_shapes, X_trn_missed = face_detection(X_trn)
X_val_shapes, X_val_missed = face_detection(X_val)
X_test_shapes, X_test_missed = face_detection(X_test)

#make a copy of each dataset to apply occlusion simulation
X_trn_occ = copy.copy(X_trn[:])
X_val_occ = copy.copy(X_val[:])
X_test_occ = copy.copy(X_test[:])

# remove intances where face detection were not successful on them
trn_lbls_occ = trn_lbls.drop(X_trn_missed).reset_index()
X_trn_occ = np.delete(X_trn_occ, X_trn_missed, axis=0)
X_trn_shapes =  np.delete(X_trn_shapes, X_trn_missed, axis=0)

val_lbls_occ = val_lbls.drop(X_val_missed).reset_index()
X_val_occ = np.delete(X_val_occ, X_val_missed, axis=0)
X_val_shapes =  np.delete(X_val_shapes, X_val_missed, axis=0)

test_lbls_occ = test_lbls.drop(X_test_missed).reset_index()
X_test_occ = np.delete(X_test_occ, X_test_missed, axis=0)
X_test_shapes =  np.delete(X_test_shapes, X_test_missed, axis=0)

test_lbls_occ= test_lbls_occ[['filename', 'usage', 'neutral', 'happiness', 'surprise',
             'sadness', 'anger', 'disgust', 'fear','contempt', 'unknown', 'NF', 'target']  ]

test_lbls_occ.to_csv('/home/ubuntu/Notebooks/Datasets/FERPlus_occ/test_label.csv', index=False)

plot_class_distribution(trn_lbls_occ['target'], 'train')
plot_class_distribution(val_lbls_occ['target'], 'validation')
plot_class_distribution(test_lbls_occ['target'], 'test')

#set VR headset dimension [w, h]
#initialize dimension based on Samsung's VR headset model Gear 
#https://www.samsung.com/global/galaxy/gear-vr/specs/
VR_dimension = [20, 10]
VR_patch(X_trn_occ, X_trn_shapes)
VR_patch(X_val_occ, X_val_shapes)
VR_patch(X_test_occ, X_test_shapes)

# create the dataset structure based on the 8 emotion classes 
base_path = '/home/ubuntu/Notebooks/Datasets/FERPlus_occ/'
dataset_structure(X_trn_occ, trn_lbls_occ, 'train', base_path)
dataset_structure(X_val_occ, val_lbls_occ, 'validation', base_path)
dataset_structure(X_test_occ, test_lbls_occ, 'test', base_path)

