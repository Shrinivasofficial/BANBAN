import numpy as np
from imageio import imread
from PIL import Image

def preprocess_input(x, v2=True):
    """Preprocesses the input image for the model."""
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(image_name):
    """Reads an image from a file."""
    return imread(image_name)

def _imresize(image_array, size):
    """Resizes an image array to the specified size."""
    return np.array(Image.fromarray(image_array).resize(size, Image.ANTIALIAS))

def to_categorical(integer_classes, num_classes=2):
    """Converts integer labels to one-hot encoding."""
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
