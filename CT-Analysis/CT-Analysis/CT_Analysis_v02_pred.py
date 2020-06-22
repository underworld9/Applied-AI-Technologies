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

BATCH_SIZE = 64
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

model = tf.keras.models.load_model('saved_model_ct/model_v07')

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

cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  return figure

plot_confusion_matrix(cm, CLASS_NAMES)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                100*np.max(predictions_array),
                                CLASS_NAMES[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_true, image_batch)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_true)
plt.show()


"""


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y_true, image_batch)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y_true)
plt.tight_layout()
plt.show()
"""