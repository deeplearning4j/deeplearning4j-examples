from scipy import ndimage
import jumpy as jp
import numpy as np
import os


path = os.path.dirname(os.path.abspath(__file__))

# load tensorflow model
tf_model = jp.TFModel(path + '/mnist.pb')

# load jpg to numpy array
image = ndimage.imread(path + '/img_1.jpg').reshape((1, 28, 28))

image = np.cast[float](image)

image /= 255.

# inference - uses nd4j
prediction = tf_model(image)  # prediction is a jumpy array

# get label from predction using argmax
label = jp.argmax(prediction.reshape((10,)))

print(label)
