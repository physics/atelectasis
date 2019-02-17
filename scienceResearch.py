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
    buffer = 8100
    train_images = []
    train_labels = []

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
    counter = 0
    othercounter = 0
    for i in range(buffer):
        othercounter = othercounter + 1
        x = random.uniform(0, 1)
        if (x < 0.5) and normalPtr < len(normal):
            try:
                # print("In normal if loop." + " " + str(
                #im_frame = Image.open("./Normal/" + normal[normalPtr])
                #np_frame = np.array(im_frame.getdata())
                print("./Normal/" + normal[normalPtr]) 
                np_frame = imageio.imread("./Normal/" + normal[normalPtr])
                np_frame = np_frame/255
                print(normalPtr, "works")
                # print("Np_frame shape: " + str(np_frame.shape))
                np_frame = cv2.resize(np_frame,(256,256))
                train_images.append(np_frame)
                train_labels.append(0)
                normalPtr = normalPtr + 1
                counter = counter + 1
                
            except:
                normalPtr = normalPtr + 1
                pass
            
        if (x > 0.5) and badPtr < len(atelectasis):
            try:
                print("not good yet")
                # print("In diseased if loop." + " " + str(x))
                #im_frame = Image.open("./Atelectasis/" + atelectasis[badPtr])
                #np_frame = np.array(im_frame.getdata())
                np_frame = imageio.imread("./Atelectasis/" + atelectasis[badPtr])
                print("good")
                #imgplot = plt.imshow(np_frame)
                #print(np_frame)
                np_frame = np_frame/255
                np_frame = cv2.resize(np_frame,(256,256))

                train_images.append(np_frame)
                train_labels.append(1)
                counter = counter + 1
                badPtr = badPtr + 1
                
            except:
                badPtr = badPtr + 1
                pass

    print(len(atelectasis))
    print(len(normal))
    print(atelectasis[3000])
    print(atelectasis[10])

    print(badPtr)
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    
    print(counter)
    print(othercounter)
    print(train_images.shape)   
    return train_images, train_labels
    # , test_images, test_labels


train_images, train_labels = process()


model = keras.Sequential([
    keras.layers.Reshape((256, 256, 1), input_shape=(256, 256)),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(256,256,1)),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer=keras.optimizers.SGD(lr=0.00001, momentum=0.9, decay=1e-6, nesterov=True),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


model.summary()
model.fit(train_images, train_labels, epochs=20)

model.save_weights('./checkpoints/my_checkpoint')

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)
