import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AvgPool2D

#Warning: "known incorrect sRGB profile" wird ausgeblendet
import warnings
warnings.filterwarnings("ignore")

#pfad festlegen. 
PATH = 'C:\\ct_scan'

train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'val')

train_dir = pathlib.Path(train_dir)
val_dir = pathlib.Path(val_dir)


train_image_count = len(list(train_dir.glob('*/*.png'))) + len(list(train_dir.glob('*/*.jpg')))
val_image_count = len(list(val_dir.glob('*/*.png'))) + len(list(val_dir.glob('*/*.jpg')))

list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
val_list_ds = tf.data.Dataset.list_files(str(val_dir/'*/*'))

#Klassen werden in nparray gespeichert
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])

print(CLASS_NAMES, train_image_count, val_image_count)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 30
STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)
validation_steps = np.ceil(val_image_count/BATCH_SIZE)

def get_label(file_path):
  #*/*.jpg ist der Klassenname. 
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor

  #This op also supports decoding PNGs and non-animated GIFs since the 
  #interface is the same, though it is cleaner to use tf.image.decode_image.
  img = tf.image.decode_jpeg(img, channels=3) 
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

def prepare_for_training(ds, shuffle_buffer_size=1000):


  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(labeled_ds)
val_ds = prepare_for_training(val_labeled_ds)

image_batch, label_batch = next(iter(train_ds))
val_image_batch, val_label_batch = next(iter(val_ds))

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()

#show_batch(image_batch.numpy(), label_batch.numpy())

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()


print("Number of layers in the base model: ", len(base_model.layers))

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

#inputs = base_model.input
#x = base_model(inputs, training=False)
#x = keras.layers.GlobalAveragePooling2D()(x)
#x = keras.layers.Dropout(0.4)(x)
#outputs = keras.layers.Dense(2)(x)
#model = keras.Model(inputs, outputs)


inputs = base_model.input
x = base_model(inputs, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256)(x)
x = keras.layers.Dense(128)(x)
x = keras.layers.Dropout(0.3)(x)
prediction = keras.layers.Dense(2)(x)
model = keras.Model(inputs, prediction)


model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 5

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    validation_data=val_ds,
                    validation_steps = validation_steps)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 10

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=val_ds,
                         steps_per_epoch = STEPS_PER_EPOCH,
                         validation_steps = validation_steps
                         )

model.save("saved_model_ct/vgg16_v01")


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
