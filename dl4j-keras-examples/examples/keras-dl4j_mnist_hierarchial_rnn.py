# Based on official keras mnist example
# Assumes that Theano would be used as a Keras model
# Uses the hijack code

from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils
from kerasdl4j import backend as dl4j

raise ValueError("TimeDistributed is unsupported by Deeplearning4j, but is under active development. Please let us know on Gitter if you have feedback about this feature.")

# Training parameters.
batch_size = 32
nb_classes = 10
nb_epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

# The data, shuffled and split between train and test sets.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshapes data to 4D for Hierarchical RNN.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Converts class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

row, col, pixel = X_train.shape[1:]

# 4D input.
x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(output_dim=row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(nb_classes, activation='softmax')(encoded_columns)
model = Model(input=x, output=prediction)

# installs the DL4J backend to your model instance
dl4j.install_dl4j_backend(model)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training.
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

# Evaluation.
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
