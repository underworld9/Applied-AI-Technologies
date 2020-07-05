import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/home/stefan/AI/CT_Scans/ct_scan'

train_dir = os.path.join(PATH, 'train')                             # set train directory
validation_dir = os.path.join(PATH, 'val')                          # set validation directory

train_normal_dir = os.path.join(train_dir, 'normal')                # normal train data
train_covid_dir = os.path.join(train_dir, 'covid')                  # covid train data
validation_normal_dir = os.path.join(validation_dir, 'normal')      # validation normal data
validation_covid_dir = os.path.join(validation_dir, 'covid')        # validation covid data

num_normal_tr = len(os.listdir(train_normal_dir))                   # getting number of normal train data
num_covid_tr = len(os.listdir(train_covid_dir))                     # getting number of covid train data

num_normal_val = len(os.listdir(validation_normal_dir))             # getting number of normal val data
num_covid_val = len(os.listdir(validation_covid_dir))               # getting number of covid val data

total_train = num_normal_tr + num_covid_tr                          # total train number
total_val = num_normal_val + num_covid_val                          # total val number

print('total training normal images:', num_normal_tr)
print('total training covid images:', num_covid_tr)

print('total validation normal images:', num_normal_val)
print('total validation covid images:', num_covid_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 20
epochs = 100
steps_per_epoch = n date
validation_steps = 22
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_image_generator = ImageDataGenerator(rescale=1./255)          # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)     # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


def createModel():
    """model = Sequential([

        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64,  kernel_size=(5, 5), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(5, 5), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(1)
    ])"""

    model = Sequential()
    """Convolutional Layer 1"""
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)))
    model.add(Dropout(0.2))
    """Convolutional Layer 2"""
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    """Convolutional Layer 3"""
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.2))

    """Flatten conv2 output"""
    model.add(Flatten())


    """Dense Layer 2"""
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))


    """Dense Layer 1"""
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5)) # Dropout

    """Output Layer"""
    model.add(Dense(1))


    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

model = createModel()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_steps
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#model.save('saved_model_ct/model_v02')

