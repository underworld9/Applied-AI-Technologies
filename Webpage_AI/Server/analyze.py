import sys
import tensorflow as tf
import os

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

PATH_TO_MODEL = os.getcwd() + "/saved_model_ct/model_v07"

BATCH_SIZE = 1
# Batch size muss >= test_image_count sein
IMG_HEIGHT = 224
IMG_WIDTH = 224

filename = ""

if not sys.argv[1]:
    print("No File given")
else:
    filename = sys.argv[1]


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor

    # This op also supports decoding PNGs and non-animated GIFs since the
    # interface is the same, though it is cleaner to use tf.image.decode_image.
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def read_image(filename):
    current = os.getcwd()
    os.chdir(os.getcwd()+"/uploads/images")
    img = tf.io.read_file(filename)
    img = decode_img(img)
    os.chdir(current)
    return img

def prepare_for_testing(img):

    img = img.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.

    return img


img = read_image(filename)

img = prepare_for_testing(img)

model = tf.keras.models.load_model(PATH_TO_MODEL)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# irgendwas fehlt mir bei dieser sache :( - Simon?

print(probability_model.predict(img))