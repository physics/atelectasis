#!/usr/bin/env python3

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image
import time

# More image manipulation libraries

import imageio

print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#creates empty array

print(tf.__version__)

def process():
    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    test_names = []
    train_names = []

    test_file = open('testing.txt', 'r')
    train_file = open('training.txt', 'r')

    for line in train_file:
      splitted = line.split(',')
      for i in splitted:
        newstr =  i.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        if "Normal" in newstr:
          try:
            np_frame = imageio.imread(newstr)
            np_frame = np_frame/255

            # print("Np_frame shape: " + str(np_frame.shape))
            np_frame = cv2.resize(np_frame,(256,256))



            train_images.append(np_frame)
            train_labels.append(0)
          except:
            print(newstr)
            pass

        if "Atelectasis" in newstr:
          try:
            np_frame = imageio.imread(newstr)
            np_frame = np_frame/255
            # print("Np_frame shape: " + str(np_frame.shape))
            np_frame = cv2.resize(np_frame,(256,256))
            train_images.append(np_frame)
            train_labels.append(1)
          except:
            print(newstr)
            pass

    for line in test_file:
      splitted = line.split(',')
      for i in splitted:
        newstr =  i.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        if "Normal" in newstr:
            try:
                np_frame = imageio.imread(newstr)
                np_frame = np_frame/255
                # print("Np_frame shape: " + str(np_frame.shape))
                np_frame = cv2.resize(np_frame,(256,256))

                test_images.append(np_frame)

                test_labels.append(0)
            except:
                print(newstr)
                pass

        if "Atelectasis" in newstr:
            try:
                np_frame = imageio.imread(newstr)
                np_frame = np_frame/255
                # print("Np_frame shape: " + str(np_frame.shape))
                np_frame = cv2.resize(np_frame,(256,256))
                test_images.append(np_frame)
                test_labels.append(1)
            except:
                print(newstr)
                pass

    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)

    test_labels = np.asarray(test_labels)
    train_labels = np.asarray(train_labels)

    return train_labels, train_images, test_labels, test_images


train_labels, train_images, test_labels, test_images = process()
''' YO THIS MODEL WORKS 72% 100 epochs
model = keras.Sequential([
   keras.layers.Reshape((256, 256, 1), input_shape=(256, 256)) ,

   keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(256,256,1)),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256,256,1)),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Flatten(),

   keras.layers.Dense(100, activation='relu') ,
   keras.layers.Dropout(0.5),

   keras.layers.Dense(1, activation='sigmoid')
])
'''

'''
model = keras.Sequential([
   keras.layers.Reshape((256, 256, 1), input_shape=(256, 256)) ,

   keras.layers.Conv2D(64, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),

   keras.layers.Conv2D(64, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),
   
   keras.layers.Conv2D(128, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),

   keras.layers.Conv2D(128, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Conv2D(256, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),

   keras.layers.Conv2D(256, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Conv2D(512, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),

   keras.layers.Conv2D(512, (5, 5), padding='same'),
   keras.layers.LeakyReLU(),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Flatten(),

   keras.layers.Dense(1024) ,
   keras.layers.LeakyReLU(),
   keras.layers.Dropout(0.5),

   keras.layers.Dense(2048),
   keras.layers.LeakyReLU(),
   keras.layers.Dropout(0.5),

   keras.layers.Dense(1, activation='sigmoid')
])
'''

model = keras.Sequential([
   keras.layers.Reshape((256, 256, 1), input_shape=(256, 256)) ,

   keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='ones'),
   keras.layers.LeakyReLU(),
   keras.layers.MaxPooling2D(pool_size=(2, 2)),

   keras.layers.Flatten(),

   keras.layers.Dense(100, kernel_initializer='ones') ,
   keras.layers.LeakyReLU(),
   keras.layers.Dropout(0.5),

   keras.layers.Dense(1, activation='sigmoid',  kernel_initializer='ones')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
model.summary()

for i in range(0,10):
  model.fit(train_images, train_labels, epochs=1)
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy for epoch ' + str((i+1)*5) + ":", str(test_acc))

model.save_weights('./checkpoints/my_checkpoint')

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)
