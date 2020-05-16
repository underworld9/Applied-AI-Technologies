import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = 'C:\\ct_scan'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_normal_dir = os.path.join(train_dir, 'normal') 
train_covid_dir = os.path.join(train_dir, 'covid') 
validation_normal_dir = os.path.join(validation_dir, 'normal')  
validation_covid_dir = os.path.join(validation_dir, 'covid')

num_normal_tr = len(os.listdir(train_normal_dir))
num_covid_tr = len(os.listdir(train_covid_dir))

num_normal_val = len(os.listdir(validation_normal_dir))
num_covid_val = len(os.listdir(validation_covid_dir))

total_train = num_normal_tr + num_covid_tr
total_val = num_normal_val + num_covid_val

print('total training normal images:', num_normal_tr)
print('total training covid images:', num_covid_tr)

print('total validation normal images:', num_normal_val)
print('total validation covid images:', num_covid_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 20
epochs = 10
steps_per_epoch = 136
validation_steps = 22
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

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
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        #Dropout(0.4),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        #Dropout(0.2),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        #Dropout(0.1),
        Flatten(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dense(1)
    ])

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

