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

#creates empty array

print(tf.__version__)

def process():
    
    buffer = 20

    train_images = np.array([])
    train_labels = np.array([])
    # testImages = []
    # testLabels = []

    normal = []
    fname = "normal.txt"
    with open(fname) as f:
        for line in f:
            current = line.replace(" ", "").replace("\n", "")
            normal.append(current)

    atelectasis = []
    fname2 = "atelectasis.txt"
    with open(fname2) as f2:
        for line in f2:
            current2 = line.replace(" ", "").replace("\n", "")
            atelectasis.append(current2)

    normalPtr = 0
    badPtr = 0

    for i in range(buffer):
        x = random.uniform(0, 1)
        if (x < 0.5) and normalPtr < len(normal):
            # print("In normal if loop." + " " + str(x))
            #im_frame = Image.open("./Normal/" + normal[normalPtr])
            #np_frame = np.array(im_frame.getdata())
            np_frame = imageio.imread("./Normal/" + normal[normalPtr] )
            np_frame = np_frame/255
            '''
            print("----123----")
            print(normal[normalPtr])
            print(np_frame)
            print(np_frame.shape)
            print("----456----")
            '''
            # print("Np_frame shape: " + str(np_frame.shape))
            np_frame = cv2.resize(np_frame,(256,256))
            train_images.append(np_frame)
            train_labels.append(0)
            normalPtr = normalPtr + 1
        if(x > 0.5) and badPtr < len(atelectasis):
            # print("In diseased if loop." + " " + str(x))
            #im_frame = Image.open("./Atelectasis/" + atelectasis[badPtr])
            #np_frame = np.array(im_frame.getdata())
            np_frame = imageio.imread("./Atelectasis/" + atelectasis[badPtr])
            # imgplot = plt.imshow(np_frame)
            #print(np_frame)
            np_frame = np_frame/255
            np_frame = cv2.resize(np_frame,(256,256))
            train_images.append(np_frame)
            train_labels.append(1)
            
            badPtr = badPtr + 1
            
        '''
        x = random.randint(0, 1)

        if (x < 0.5) and normalPtr < len(normal):
            im_frame = Image.open("./Normal/" + normal[normalPtr])
            np_frame = np.array(im_frame.getdata())
            np_frame = np_frame/255
            np_frame = cv2.resize(np_frame,(256,256))
            testImages.append(np_frame)
            testLabels.append(0)
            normalPtr = normalPtr + 1
        if (x > 0.5) and badPtr < len(atelectasis):
            im_frame = Image.open("./Atelectasis/" + atelectasis[badPtr])
            np_frame = np.array(im_frame.getdata())
            np_frame = np_frame/255
            np_frame = cv2.resize(np_frame,(256,256))
            testImages.append(np_frame)
            testLabels.append(1)
            badPtr = badPtr + 1
        '''

    # test_images = np.asarray(testImages)
    # test_labels = np.asarray(testLabels)
    print(train_images.shape)
    return train_images, train_labels
    # , test_images, test_labels

train_images, train_labels = process()
# , test_images, test_labels 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(256, 256)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()
model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)
