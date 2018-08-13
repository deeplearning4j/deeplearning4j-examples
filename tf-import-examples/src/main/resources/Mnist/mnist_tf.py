import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.datasets import mnist
from PIL import Image
import numpy as np

# Load data
train_data, test_data = mnist.load_data()
x_train, y_train = train_data
x_test, y_test = test_data

# save some images for testing from java
for i in range(10):
    img_file = 'images/img_{}.jpg'.format(i + 1)
    img = Image.fromarray(test_data[0][i])
    img.save(img_file)

# Normalize
x_train = x_train / 255.0
x_test  = x_test / 255.0


weights = None


def get_model(training):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation='relu'))
    if training:
        model.add(Dropout(0.5))
    else:
        model.add(Lambda(lambda x: x + 0.))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train():
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = get_model(True)
        model.fit(x_train, y_train, epochs=5)
        global weights
        weights = model.get_weights()

def save():
    # save model to a protobuff
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = get_model(False)
        model.set_weights(weights)
        model.evaluate(x_test, y_test)
        sess.run(tf.global_variables_initializer())
        output_node_name = model.output.name.split(':')[0]
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_node_name])
        with tf.gfile.GFile("mnist.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

train()
save()
