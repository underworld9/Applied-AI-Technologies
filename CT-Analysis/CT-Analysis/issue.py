import tensorflow as tf

#number of parallel calls is set dynamically based on available CPU. 
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import itertools
import tensorflow.keras.backend as K
import cv2



from PIL import Image
from tensorflow import keras

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Warning: "known incorrect sRGB profile" wird ausgeblendet
#Manche Bilder haben eine Farbtiefe von 32bit
import warnings
warnings.filterwarnings("ignore")

PATH = 'C:\\ct_scan'

test_dir = os.path.join(PATH, 'test')

test_dir = pathlib.Path(test_dir)


test_image_count = len(list(test_dir.glob('*/*.png'))) + len(list(test_dir.glob('*/*.jpg')))

list_ds = tf.data.Dataset.list_files(str(test_dir/'*/*'))

#Klassen werden in nparray gespeichert
CLASS_NAMES = np.array([item.name for item in test_dir.glob('*')])

print(CLASS_NAMES, "Anzahl Bilder: " , test_image_count)

BATCH_SIZE = 70
#Batch size muss >= test_image_count sein
IMG_HEIGHT = 224
IMG_WIDTH = 224

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

def prepare_for_testing(ds):

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

test_ds = prepare_for_testing(labeled_ds)

image_batch, label_batch = next(iter(test_ds))

model = tf.keras.models.load_model('saved_model_ct/vgg16_v02')

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(image_batch)
 
def getytrue(label_batch):
    #bringt das label batch in einen 1d array mit 0 für covid und 1 für normal
    scalar = []
    for i in range(test_image_count):
        if label_batch[i][0] == [True]:
            scalar.append(0)
        else: 
            scalar.append(1)
    return scalar

def getypred(predictions):
    #gibt die stelle des maxwertes in predictions zurück.
    pred = []
    for i in range(test_image_count):
        pred.append(np.argmax(predictions[i]))

    return pred

y_true = getytrue(label_batch)
y_pred = getypred(predictions)

def plotheatmap(convlayer):
    n = 1

    img = image_batch[n]

    x = tf.expand_dims(img, 0)

    def getsinglepred(predictions):
        #gibt die stelle des maxwertes in predictions in einem array zurück.
        pred = []
        for i in range(1):
            pred.append(np.argmax(predictions[i]))
            print(CLASS_NAMES[pred])

        return pred

    preds = probability_model.predict(x)

    print(getsinglepred(preds))

    #model.summary()

    if getsinglepred(preds) == [0] or getsinglepred(preds) == [1]:

        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer("vgg16").get_layer(convlayer)
            iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(x)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((9, 9))
        plt.matshow(heatmap)
        plt.show()

        INTENSITY = 0.5

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img = heatmap * INTENSITY + img

        plt.matshow(img)
        plt.show()
    else:
        plt.matshow(img)
        plt.show()

plotheatmap("block5_conv3")